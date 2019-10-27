from typing import Optional, Callable
import numpy as np

from numba import njit, prange
import numba as nb



@njit(parallel=True)
def col2im(col, x_shape, n_filter, filter_height, filter_width, padding, stride):
    nimages, input_channels, height, width = x_shape
    padded_height = height+2*padding
    padded_width = width+2*padding

    output_h = int((padded_height-filter_height) // stride) + 1
    output_w = int((padded_width-filter_width) // stride) + 1
    patch_shape = (input_channels, filter_height, filter_width)
    x_padded = np.zeros(
                (nimages, input_channels, padded_height, padded_width))
    x_padded = np.zeros(nimages, input_channels*padded_height*padded_width)
    # Reshape col into original shaped as defined in im2col
    col_reshaped = col.reshape(input_channels*filter_height*filter_width, -1, nimages)
    col_reshaped = col_reshaped.transpose(2, 0, 1)

    # Reverse the im2col process
    for n in prange(nimages):
        for i in prange(output_h):
           for j in prange(output_w):
               for c in prange(output_channels)
               start_h = i*stride
               end_h = start_h+filter_height
               start_w = j*stride
               end_w = start_w+filter_width
               patch = col_reshaped[n, :, j+i*output_w].copy()
               # x_padded[n, :, start_h:end_h, start_w:end_w]+=patch.reshape(patch_shape)
               x_padded[n, :, start_h:end_h, start_w:end_w]+=patch
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

@njit(parallel=True)
def test(x):
    n = x.shape[0]
    a = np.sin(x)
    b = np.cos(a * a)
    acc = 0
    for i in prange(n - 2):
        for j in prange(n - 1):
            acc += b[i] + b[j + 1]
    return acc

padding = 0
stride = 1
filter_height = 3
filter_width = 3
x_shape = (2, 2, 10, 10)
col = np.zeros((filter_height*filter_height*2,  4*2))
col2im(col, x_shape, filter_height, filter_width, padding, stride)

col2im.parallel_diagnostics(level=4)
