from typing import Optional, Callable
import numpy as np

from numba import njit, prange
import numba as nb

from nn import Parameter
from .layer import Layer


class ConvLayerProNumba(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayerProNumba, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.cache = None
        self.initialize()

    def forward(self, data):
        padding, stride = self.padding, self.stride

        W = self.weight.data.transpose(1, 0, 2, 3)
        n_filter, nchannel, filter_h, filter_w = W.shape
        nimages, nchannel, height, width = data.shape
        output_h = int(height - filter_h + 2*padding) // stride + 1
        output_w = int(width - filter_w + 2*padding) // stride + 1

        # X_col = input_channels*filter_height*filter_width, output_h*output_w*nimages
        X_col = self.im2col(data, filter_h, filter_w, padding, stride)

        # W = n_filter, input_channels*filter_height*filter_width
        W_col = W.reshape(n_filter, -1)
        b = np.tile(self.bias.data, (X_col.shape[-1], 1)).T
        out = W_col @ X_col + b
        out = out.reshape(n_filter, output_h, output_w, nimages)
        out = out.transpose(3, 0, 1, 2)

        # Note that we save tranposed W
        self.cache = (data, W, X_col)
        return out

    # def backward(self, previous_partial_gradient):
    #     X, W, X_col = self.cache
    #
    #     stride, padding = self.stride, self.padding
    #     n_filter, nchannel, h_filter, w_filter = W.shape
    #
    #     db = np.sum(previous_partial_gradient, axis=(0, 2, 3))
    #     dy = previous_partial_gradient.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    #
    #     dkernel = dy @ X_col.T
    #     dkernel = dkernel.reshape(W.shape)
    #     dkernel = dkernel.transpose(1, 0, 2, 3)
    #
    #     W_col = W.reshape(n_filter, -1)
    #
    #     #dX_col = input_channels*filter_height*filter_width, output_h*output_w*nimages
    #     dX_col = W_col.T @ dy
    #
    #     dinput = self.col2im(dX_col, X.shape, n_filter, h_filter, w_filter, padding, stride)
    #     # dinput_ref = self.col2im_indices(dX_col, X.shape, h_filter, w_filter, padding, stride)
    #
    #     self.weight.grad = dkernel
    #     self.bias.grad = db
    #     return dinput

    def im2col(self, x, filter_height, filter_width, padding, stride):
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        nimages, input_channels, padded_height, padded_width = x_padded.shape
        output_h = int((padded_height-filter_height) // stride) + 1
        output_w = int((padded_width-filter_width) // stride) + 1

        col = np.zeros((nimages, input_channels*filter_height*filter_width,
                        output_h*output_w))

        for n in range(nimages):
            for i in range(output_h):
               for j in range(output_w):
                   start_h = i*stride
                   end_h = start_h+filter_height
                   start_w = j*stride
                   end_w = start_w+filter_width
                   patch = x_padded[n, :, start_h:end_h, start_w:end_w]
                   col[n, :, j+i*output_w] = patch.flatten()

        col = col.transpose(1 , 2, 0).reshape(filter_height*filter_width*input_channels, -1)
        return col

    # def col2im(self, col, x_shape, filter_height, filter_width, padding, stride):
    #     nimages, input_channels, height, width = x_shape
    #     padded_height = height+2*padding
    #     padded_width = width+2*padding
    #
    #     output_h = int((padded_height-filter_height) // stride) + 1
    #     output_w = int((padded_width-filter_width) // stride) + 1
    #     patch_shape = (input_channels, filter_height, filter_width)
    #     x_padded = np.zeros(
    #                 (nimages, input_channels, padded_height, padded_width),
    #                 dtype=col.dtype)
    #
    #     # Reshape col into original shaped as defined in im2col
    #     col_reshaped = col.reshape(input_channels*filter_height*filter_width, -1, nimages)
    #     col_reshaped = col_reshaped.transpose(2, 0, 1)
    #
    #     # Reverse the im2col process
    #     for n in range(nimages):
    #         for i in range(output_h):
    #            for j in range(output_w):
    #                start_h = i*stride
    #                end_h = start_h+filter_height
    #                start_w = j*stride
    #                end_w = start_w+filter_width
    #                patch = col_reshaped[n, :, j+i*output_w]
    #                x_padded[n, :, start_h:end_h, start_w:end_w]+=patch.reshape(patch_shape)
    #     if padding == 0:
    #         return x_padded
    #     return x_padded[:, :, padding:-padding, padding:-padding]

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, original_shape, kernel, padding, stride):
        nimages, output_channels, output_height, output_width = previous_grad.shape
        nimages, input_channels, height, width = original_shape
        input_channels, output_channels, filter_height, filter_width = kernel.shape

        dinput = np.zeros((nimages, input_channels, height+2*padding, width+2*padding))
        dkernel = np.zeros((input_channels*filter_height*filter_width, output_channels))
        db = np.zeros((output_channels))

        # Parallization problem
        # https://numba.pydata.org/numba-doc/latest/user/parallel.html
        # for c in range(output_channels):
        #     dkernel_reference = dkernel[:,c]
        #     db[c] = previous_grad[:,c,:,:].sum()
        #     for n in prange(nimages):
        #         ct = colmat_transposed[n, :, :].copy() # To convert to C order
                # A  = np.dot(ct, previous_grad[n,c,:,:].flatten().reshape(-1,1))
                # dkernel_reference += A.flatten()

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
        X, W, X_col = self.cache
        original_shape = X.shape
        kernel_shape = self.weight.data.shape

        # Backprop for convolution
        dinput, dkernel, db = self.backward_numba(previous_partial_gradient,
         original_shape, self.weight.data, padding, stride)

        dkernel = dkernel.reshape(kernel_shape[0], kernel_shape[2], kernel_shape[3], kernel_shape[1])
        dkernel = np.moveaxis(dkernel, -1, 1)


        # Using quick method
        db = np.sum(previous_partial_gradient, axis=(0, 2, 3))

        dy = previous_partial_gradient.transpose(1, 2, 3, 0).reshape(kernel_shape[1], -1)
        dkernel = dy @ X_col.T
        dkernel = dkernel.reshape(W.shape)
        dkernel = dkernel.transpose(1, 0, 2, 3)


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
        super(ConvLayerProNumba, self).initialize()
