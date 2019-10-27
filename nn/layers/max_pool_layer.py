import numbers

import numpy as np
from numba import njit, prange

from .layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.padded_input = None
        # self.grad = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(padded_data, filter_shape, padding, stride):
        nimages, input_channels, padded_height, padded_width = padded_data.shape
        input_channels, output_channels, filter_height, filter_width = filter_shape
        output_height = int((padded_height-filter_height) // stride) + 1
        output_width = int((padded_width-filter_width) // stride) + 1

        output = np.zeros((nimages, output_channels, output_height, output_width))

        for n in prange(nimages):
            for i in prange(output_height):
               for j in prange(output_width):
                   start_h = i*stride
                   end_h = start_h+filter_height
                   start_w = j*stride
                   end_w = start_w+filter_width
                   patch = padded_data[n, :, start_h:end_h, start_w:end_w].copy()
                   for c in prange(output_channels):
                       output[n, c, i, j] = np.max(patch[c])

        return output

    def forward(self, data):
        # TODO

        # Declare variables
        padding, p = self.padding, self.padding
        stride = self.stride
        filter_shape = (data.shape[1], data.shape[1], self.kernel_size, self.kernel_size)
        padded_data = np.pad(data, ((0,0),(0,0),(p,p),(p,p)),
                        'constant',constant_values=(0))

        output = self.forward_numba(padded_data, filter_shape, padding, stride)
        self.padded_input = padded_data
        return output


    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, padded_data, filter_shape, padding, stride):
        # data is N x C x H x W
        # TODO
        nimages, output_channels, output_height, output_width = previous_grad.shape
        nimages, input_channels, height, width = padded_data.shape
        input_channels, output_channels, filter_height, filter_width = filter_shape

        dinput = np.zeros((padded_data.shape))
        for n in prange(nimages):
                for i in prange(output_height):
                   for j in prange(output_width):
                       start_h = i*stride
                       end_h = start_h+filter_height
                       start_w = j*stride
                       end_w = start_w+filter_width
                       patch = padded_data[n, :, start_h:end_h, start_w:end_w].copy()
                       for c in prange(output_channels):
                           max = (patch[c,:,:]==np.max(patch[c,:,:]))
                           dinput[n, c, start_h:end_h, start_w:end_w]+=previous_grad[n,c,i,j]*max

        return dinput

    def backward(self, previous_partial_gradient):
        # TODO
        padding = self.padding
        stride = self.stride
        nimages, nchannels, padded_width, padded_height = self.padded_input.shape
        filter_shape = (nchannels, nchannels, self.kernel_size, self.kernel_size)

        dinput = self.backward_numba(previous_partial_gradient,
         self.padded_input, filter_shape, padding, stride)

        if (padding>0):
            # self.grad = dinput[:,:,padding:-padding,padding:-padding]

            return dinput[:,:,padding:-padding,padding:-padding]
        else:
            # self.grad = dinput
            return dinput

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))
