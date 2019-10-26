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
        self.input_shape = None
        self.colmat_transposed = None
        self.initialize()

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(padded_data, weight, bias, stride):
        nimages, input_channels, padded_height, padded_width = padded_data.shape
        input_channels, output_channels, filter_height, filter_width = weight.shape
        output_height = int((padded_height-filter_height) // stride) + 1
        output_width = int((padded_width-filter_width) // stride) + 1

        output = np.zeros((nimages, output_channels, output_height, output_width))
        colmat_transposed = np.zeros((nimages, input_channels*filter_height*filter_width, output_height*output_width))

        for n in prange(nimages):
            for i in prange(output_height):
               for j in prange(output_width):
                   start_h = i*stride
                   end_h = start_h+filter_height
                   start_w = j*stride
                   end_w = start_w+filter_width
                   patch = padded_data[n, :, start_h:end_h, start_w:end_w]
                   colmat_transposed[n, :, i*output_width+j] = patch.flatten()
                   for c in prange(output_channels):
                       xw = np.multiply(patch, weight[:,c,:,:])
                       output[n, c, i, j] = np.sum(xw) + bias[c]*output_width*output_height

        return (output, colmat_transposed)

    def forward(self, data):
        # Declare variables
        p = self.padding
        stride = self.stride
        kernel_size = self.kernel_size
        weight = self.weight.data
        bias = self.bias.data
        padded_data = np.pad(data, ((0,0),(0,0),(p,p),(p,p)),
                        'constant',constant_values=(0))
        output, self.colmat_transposed = self.forward_numba(padded_data,
                                            weight, bias, stride)
        self.input_shape = data.shape
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, colmat_transposed, original_shape, kernel, padding, stride):
        nimages, output_channels, output_height, output_width = previous_grad.shape
        nimages, input_channels, height, width = original_shape
        input_channels, output_channels, filter_height, filter_width = kernel.shape

        dinput = np.zeros((nimages, input_channels, height+2*padding, width+2*padding))
        dkernel = np.zeros((input_channels*filter_height*filter_width, output_channels))
        db = np.zeros((output_channels))

        # Parallization problem
        # https://numba.pydata.org/numba-doc/latest/user/parallel.html
        for c in range(output_channels):
            dkernel_reference = dkernel[:,c]
            db[c] = previous_grad[:,c,:,:].sum()
            for n in prange(nimages):
                ct = colmat_transposed[n, :, :].copy() # To convert to C order
                A  = np.dot(ct, previous_grad[n,c,:,:].flatten().reshape(-1,1))
                dkernel_reference += A.flatten()

        for n in prange(nimages):
            for c in range(output_channels):
                for i in prange(output_height):
                   for j in range(output_width):
                       start_h = i*stride
                       end_h = start_h+filter_height
                       start_w = j*stride
                       end_w = start_w+filter_width
                       dinput[n, :, start_h:end_h, start_w:end_w]+=previous_grad[n,c,i,j]*kernel[:,c,:,:]

        return (dinput, dkernel, db)

    def backward(self, previous_partial_gradient):
        padding, p = self.padding, self.padding
        stride = self.stride
        kernel_size = self.kernel_size
        original_shape = self.input_shape
        kernel_shape = self.weight.data.shape

        dinput, dkernel, db = self.backward_numba(previous_partial_gradient,
         self.colmat_transposed, original_shape, self.weight.data, padding, stride)

        dkernel = dkernel.reshape(kernel_shape[0], kernel_shape[2], kernel_shape[3], kernel_shape[1])
        dkernel = np.moveaxis(dkernel, -1, 1)

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
