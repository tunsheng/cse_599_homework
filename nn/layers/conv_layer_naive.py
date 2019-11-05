from typing import Optional, Callable
import numpy as np

from numba import njit, prange
import numba as nb

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.cache = None
        self.initialize()

    def forward(self, data):
        """
        Naive implementation of convolution
        more memory friendly
        """
        padding, p, stride = self.padding, self.padding, self.stride
        weight, bias = self.weight.data, self.bias.data
        padded_data = np.pad(data, ((0,0),(0,0),(p,p),(p,p)),
                        'constant',constant_values=(0))
        # To improve speed, reshape weight
        # kernel = np.moveaxis(self.weight.data, 1, 0)
        output = self.forward_numba(padded_data, weight, bias, stride)

        self.cache = (padded_data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(padded_data, kernel, bias, stride):
        nimages, input_channels, padded_height, padded_width = padded_data.shape
        # output_channels, input_channels, filter_height, filter_width = kernel.shape
        input_channels, output_channels, filter_height, filter_width = kernel.shape
        output_height = int((padded_height-filter_height) // stride) + 1
        output_width = int((padded_width-filter_width) // stride) + 1

        output = np.zeros((nimages, output_channels, output_height, output_width))

        for n in prange(nimages):
            for c in prange(output_channels):
                for i in prange(output_height):
                   for j in prange(output_width):
                       for cold in range(input_channels):
                           for kh in range(filter_height):
                               for kw in range(filter_width):
                                   sample_h = i*stride + kh
                                   sample_w = j*stride + kw
                                   patch = padded_data[n, cold, sample_h, sample_w]
                                   output[n, c, i, j] += patch*kernel[cold,c,kh,kw]+bias[c]
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, padded_data, kernel, padding, stride):
        nimages, output_channels, output_height, output_width = previous_grad.shape
        input_channels, output_channels, filter_height, filter_width = kernel.shape

        dinput = np.zeros(padded_data.shape)
        dkernel = np.zeros(kernel.shape)
        # Numba does not support slicing when accumulating
        for cold in prange(input_channels):
            for n in range(nimages):
                for c in range(output_channels):
                    for i in range(output_height):
                       for j in range(output_width):
                           for kh in range(filter_height):
                               for kw in range(filter_width):
                                   sample_h = i*stride + kh
                                   sample_w = j*stride + kw
                                   dinput[n,cold,sample_h,sample_w]+=previous_grad[n,c,i,j]*kernel[cold,c,kh,kw]
                                   dkernel[cold,c,kh,kw] += padded_data[n,cold,sample_h,sample_w] * previous_grad[n,c,i,j]

        return (dinput, dkernel)

    def backward(self, previous_partial_gradient):
        padding, p = self.padding, self.padding
        stride, kernel_size = self.stride, self.kernel_size
        padded_data = self.cache

        # Backprop for convolution
        dinput, dkernel = self.backward_numba(previous_partial_gradient,
         padded_data, self.weight.data, padding, stride)

        # Using quick method
        db = np.sum(previous_partial_gradient, axis=(0, 2, 3))

        # Update gradient
        self.weight.grad = dkernel
        self.bias.grad = db

        if (padding>0):
            return dinput[:,:,p:-p,p:-p]
        else:
            return dinput

    def selfstr(self):
        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2],
            self.weight.data.shape[3],
            self.weight.data.shape[0],
            self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(ConvLayer, self).initialize()
