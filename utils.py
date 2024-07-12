from typing import Tuple

def make_block_bb(x_min: int, y_min: int, x_max : int, y_max : int, block_size) -> Tuple[int,int,int,int]:
        

    #make the size of the BBOX, in both dimensions, a multiple of block_size
    x_size = x_max-x_min
    x_mod = x_size % block_size
    x_div = x_size // block_size
    x_add = (block_size - x_mod)
    x_add_half =  x_add // 2

    y_size = y_max-y_min
    y_mod = y_size % block_size
    y_div = y_size // block_size
    y_add = (block_size - y_mod)
    y_add_half = y_add // 2

    x_min -= x_add_half
    x_max += x_add - x_add_half
    y_min -= y_add_half
    y_max += y_add - y_add_half
    #the additions are weird in order to handle odd numbers of x/y_add

    return (x_min, y_min, x_max, y_max)

import skimage.morphology as skim
import numpy as np

def fill_with_convex_hull(array_to_fill):
    array_ch = skim.convex_hull_image(array_to_fill)
    array_invert = np.invert(array_to_fill)
    array_solid = np.logical_and(array_invert,array_ch)
    array_filled = np.logical_or(array_to_fill,array_solid)
    return array_filled