from pathlib import Path
from collections import namedtuple

import PIL.Image
import PIL.ImageShow
import numpy as np
from pyometiff import OMETIFFWriter
import webknossos as wk

import dask.array as da

import pandas as pd
from skimage.measure import label, regionprops_table
from skimage.color import label2rgb

from webknossos import BoundingBox
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
#matplotlib.use('Agg')

from webknossos_utils import Pixel_size, Annotation
import napari
from utils import fill_with_convex_hull

AUTH_TOKEN = "S-QRDIegZYX0IM1lXmyiJg" #2024-07-04
WK_TIMEOUT="3600" # in seconds
ORG_ID = "83d574429f8bc523" # gisela's webknossos

#sample1   = Annotation("6644c04d0100004a01fa11af", "deprecated","60_2_5R")
#sample2    = Annotation("664316880100008a049e890e", "deprecated","60_2_3L")
id_1 = "6644c04d0100004a01fa11af"
id_2 = "664316880100008a049e890e"
id_3 = "664606440100005102550210"

wk_id_list = [id_1,id_2,id_3]

#666c4ae70100002b015e2344
# the dataset url comes from the WEBKNOSSOS website, open the image of interest from the dashboard and check
# I removed the view information


SHOW_IMAGES = False
CLEAR_OUTPUT_DIR = True

import tifffile
import os
import glob
path = str(os.getcwd()) + "/data/train_images/"
target_path = path + "/targets/"
input_path = path + "/inputs/"

from pathlib import Path
Path(path).mkdir(parents=True, exist_ok=True)
Path(target_path).mkdir(parents=True, exist_ok=True)
Path(input_path).mkdir(parents=True, exist_ok=True)

if CLEAR_OUTPUT_DIR:
    files = glob.glob(target_path+"/*.tiff")
    files += (glob.glob(input_path+"/*.tiff"))
    for f in files:
        os.remove(f)

for wkid in wk_id_list:

    ANNOTATION_ID = wkid

    with wk.webknossos_context(token=AUTH_TOKEN):
        annotations = wk.Annotation.open_as_remote_dataset(annotation_id_or_url=ANNOTATION_ID)
        lbl_layers = annotations.get_segmentation_layers()

        label_indices = {i.name : l for l,i in enumerate(lbl_layers)}

        DATASET_NAME = annotations._properties.id['name']
        ds = wk.Dataset.open_remote(dataset_name_or_url=DATASET_NAME, organization_id=ORG_ID)
        img_layer = ds.get_color_layers()
        assert len(img_layer) == 1, "more than an image, this is unexpected for this project"
        img_layer = img_layer[0]    

        voxel_size = ds.voxel_size
        mag_list = list(img_layer.mags.keys())
        
        #print(mag_list)
        MAG = mag_list[3]
        pSize = Pixel_size(voxel_size[0] * MAG.x, voxel_size[1] * MAG.y, voxel_size[2] * MAG.z, MAG=MAG, unit="nm")

        img_data = img_layer.get_mag(pSize.MAG).read()
        lbl_data = lbl_layers[label_indices["Myelin"]].get_mag(pSize.MAG).read()

    #print(img_data.shape)
    #img_dask = da.from_array(np.swapaxes(img_data,-1,-3), chunks=(1,1,512,512))
    #print(img_dask.shape)
    # img_dask


    unique_lbls = np.unique(lbl_data)

    if np.max(unique_lbls) < 512:
        lbl_data = lbl_data.astype(np.uint8)

    lbl_dask = da.from_array(np.swapaxes(lbl_data,-1,-3), chunks=(1,5,512,512))

    segmentation = np.nonzero(lbl_dask[0,0])

    bbox = 0, 0, 0, 0
    #if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))

    block_size = 160 #size of resulting training images in pixels (both x and y)

    from utils import make_block_bb

    large_bbox = make_block_bb(x_min,y_min,x_max, y_max,block_size)

    #construct WK bbox from large_bbox
    wk_bbox = wk.BoundingBox(topleft=(x_min,y_min,0), size=(x_max-x_min,y_max-y_min,1))
    wk_bbox = wk_bbox.align_with_mag(pSize.MAG,ceil=True)
    wk_bbox = wk_bbox.from_mag_to_mag1(pSize.MAG)

    with wk.webknossos_context(token=AUTH_TOKEN, timeout=WK_TIMEOUT):
        img_data_small = img_layer.get_finest_mag().read(absolute_bounding_box=wk_bbox)
        lbl_data_small = lbl_layers[label_indices["Myelin"]].get_finest_mag().read(absolute_bounding_box=wk_bbox)
    

    lbl_dask_small = da.from_array(np.swapaxes(lbl_data_small,-1,-3), chunks=(1,2,512,512))
    img_dask_small = da.from_array(np.swapaxes(img_data_small,-1,-3), chunks=(1,2,512,512))

    from matplotlib.patches import Rectangle
    from PIL import Image
    ax = plt.gca()

    #get all annotations as bboxes

    import myelin_morphometrics as morpho
    from webknossos_utils import skibbox2wkbbox
    from scipy import ndimage

    properties = ['label', 'bbox', 'centroid']
    label_img = lbl_dask[0,0,:,:].compute()
    reg_table = regionprops_table(label_image=label_img,
                            properties=properties)
    reg_table = pd.DataFrame(reg_table)

    for index, row in reg_table.iterrows():
        obj_idx = row['label']

        bbox = skibbox2wkbbox(row.to_dict(), pSize)
        print(f"bbox size: {bbox.size}")
 
        myelin_lbl = lbl_layers[label_indices["Myelin"]].get_finest_mag().read(absolute_offset=bbox.topleft, size=bbox.size).squeeze()
        myelin_lbl = np.swapaxes(myelin_lbl,0,1)

        myelin_bw = morpho.get_BW_from_lbl(myelin_lbl, obj_idx)
        # clean myelin label map, in case other neurons are close by
        myelin_lbl[np.logical_not(myelin_bw)] = 0
        # Create the padded myelin_bw image to avoid edge effects in the contours
        myelin_bw_fill = ndimage.binary_fill_holes(myelin_bw)

        com = ndimage.center_of_mass(myelin_bw_fill)
        #check if "center of mass" index is true. If it is it is considered filled,
        #otherwise we fill it below
        expected_filled = myelin_bw_fill[int(com[0]), int(com[1])]

        #from utils import fill_with_convex_hull
        if not expected_filled:
            myelin_bw_fill = fill_with_convex_hull(myelin_bw_fill)

        myelin_lbl[np.nonzero(myelin_bw_fill)] = np.random.randint(low=1,high=255)
    
        tx = bbox.topleft[0]
        ty = bbox.topleft[1]
        wx = bbox.size.x
        wy = bbox.size.y

        xc = tx-bx
        yc = ty-by
        xwd = xc+wx
        ywd = yc+wy

        print(f"Running index: {index}")  

        try:
            lbl_dask_small[0,0,yc:ywd,xc:xwd] = np.where(myelin_lbl != 0, myelin_lbl, lbl_dask_small[0,0,yc:ywd,xc:xwd])
        except:
            print(f"    *** Index: {index} failed ***")  
            continue

    


    train_img_size = 500
    min_labels_per_image = 5

    img_x_div = bw // train_img_size
    img_y_div = bh // train_img_size
    lbl_dask_cropped = lbl_dask_small[0,0:img_y_div*train_img_size,0:img_x_div*train_img_size].compute()
    img_dask_cropped = img_dask_small[0,0:img_y_div*train_img_size,0:img_x_div*train_img_size].compute()

    import skimage.util

    idx = 0
    for y in range(img_y_div):
        for x in range(img_x_div):
            print(f"x: {x}, y: {y}")
            start_x = x * train_img_size
            start_y = y * train_img_size
            end_x = start_x + train_img_size
            end_y = start_y + train_img_size
            active_chunk = lbl_dask_cropped[0,start_y:end_y,start_x:end_x]
            nun = len(np.unique(active_chunk))
            print(f"unique elems: {nun}") 
            if nun > min_labels_per_image : #consider only chunks where we have "enough" data to be useful
                print(f"using chunk {idx}")

                #generate more data from single data by flipping, rotating
                active_flip_ud = np.flipud(active_chunk)
                active_flip_lr = np.fliplr(active_chunk)
                active_rot = np.rot90(active_chunk)

                #generate more data from single data by flipping, rotating and gauissian noise
                img_chunk = img_dask_cropped[0,start_y:end_y,start_x:end_x]
                img_flip_ud = np.flipud(img_chunk)
                img_flip_lr = np.fliplr(img_chunk)
                img_rot = skimage.util.random_noise(np.rot90(img_chunk),mode='gaussian')

                #save images of labels
                tifffile.imsave(target_path + str(ANNOTATION_ID) + "_" + str(idx) + '.tiff',
                                active_chunk.astype(np.uint8))
                # tifffile.imsave(target_path + str(ANNOTATION_ID) + "_fud__target_" + str(idx) + '.tiff',
                #                 active_flip_ud.astype(np.uint8))
                # tifffile.imsave(target_path + str(ANNOTATION_ID) + "_flr__target_" + str(idx) + '.tiff',
                #                 active_flip_lr.astype(np.uint8))
                # tifffile.imsave(target_path + str(ANNOTATION_ID) + "_rot_gauss_target_" + str(idx) + '.tiff',
                #                 active_rot.astype(np.uint8))


                #save corresponding chunk of image
                tifffile.imsave(input_path + str(ANNOTATION_ID) + "_" + str(idx) + '.tiff',img_chunk)
                #tifffile.imsave(input_path + str(ANNOTATION_ID) + "_fud_input_" + str(idx) + '.tiff',img_flip_ud)
                #tifffile.imsave(input_path + str(ANNOTATION_ID) + "_flr_input_" + str(idx) + '.tiff',img_flip_lr)
                #tifffile.imsave(input_path + str(ANNOTATION_ID) + "_rot_gauss_input_" + str(idx) + '.tiff',img_rot)
                if SHOW_IMAGES:
                    f, axarr = plt.subplots(2,4, figsize=(10,5))
                    axarr[0,0].imshow(img_chunk)
                    axarr[0,1].imshow(img_flip_ud)
                    axarr[0,2].imshow(img_flip_lr)
                    axarr[0,3].imshow(img_rot)

                    axarr[1,0].imshow(active_chunk)
                    axarr[1,1].imshow(active_flip_ud)
                    axarr[1,2].imshow(active_flip_lr)
                    axarr[1,3].imshow(active_rot)
                    for ax in axarr.flat:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()

                idx+=1
            print("")

print("Done!")

