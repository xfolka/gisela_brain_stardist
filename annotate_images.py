
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib as plt
#matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import os
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

np.random.seed(42)
lbl_cmap = random_label_cmap()

path = str(os.getcwd()) + "/data/"
to_annotate_path = path + "/to_annotate/"

X = sorted(glob(to_annotate_path + '*.tiff'))

#X = list(map(imread,X))
axis_norm = (0,1)   # normalize channels independently

model = StarDist2D(None, name='stardist', basedir='models')

for x in X:
    x = imread(x)
    x = normalize(x,1,99.8,axis=axis_norm)
    labels, details = model.predict_instances(x)

    plt.figure(figsize=(8,8))
    plt.imshow(x,clim=(0,1), cmap='gray')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.3)
    plt.axis('off')
    plt.show()


