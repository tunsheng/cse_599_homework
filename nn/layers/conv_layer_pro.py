from typing import Optional, Callable
import numpy as np

from numba import njit, prange
import numba as nb

from nn import Parameter
from .layer import Layer


class ConvLayerPro(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayerPro, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.cache = None
        self.initialize()

    def forward(self, data):
        # Declare variables
        padding = self.padding
        stride = self.stride
        kernel_size = self.kernel_size
        weight = self.weight.data
        bias = self.bias.data

        W = self.weight.data.transpose(1, 0, 2, 3)
        X = data
        n_filter, nchannel, h_filter, w_filter = W.shape
        nimages, nchannel, h_x, w_x = X.shape
        h_out = int(h_x - h_filter + 2*padding) // stride + 1
        w_out = int(w_x - w_filter + 2*padding) // stride + 1

        X_col = self.im2col_indices(X, h_filter, w_filter, padding, stride)
        W_col = W.reshape(n_filter, -1)
        b = np.tile(bias, (X_col.shape[-1],1)).T
        out = W_col @ X_col + b
        out = out.reshape(n_filter, h_out, w_out, nimages)
        out = out.transpose(3, 0, 1, 2)

        self.cache = (X, W, X_col)

        return out

    def backward(self, previous_partial_gradient):
        X, W, X_col = self.cache
        stride, padding = self.stride, self.padding
        n_filter, nchannel, h_filter, w_filter = W.shape

        db = np.sum(previous_partial_gradient, axis=(0, 2, 3))

        dy = previous_partial_gradient.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dkernel = dy @ X_col.T
        dkernel = dkernel.reshape(W.shape)
        dkernel = dkernel.transpose(1, 0, 2, 3)

        W_col = W.reshape(n_filter, -1)
        dX_col = W_col.T @ dy
        dinput = self.col2im_indices(dX_col, X.shape, h_filter, w_filter, padding, stride)

        self.weight.grad = dkernel
        self.bias.grad = db
        return dinput



    def get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        # assert (H + 2 * padding - field_height) % stride == 0
        # assert (W + 2 * padding - field_height) % stride == 0
        out_height = int(H + 2 * padding - field_height) // stride + 1
        out_width = int(W + 2 * padding - field_width) // stride + 1

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return (k.astype(int), i.astype(int), j.astype(int))


    def im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_im2col_indices(x.shape, field_height, field_width, padding, stride)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols


    def col2im_indices(self, cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
        """ An implementation of col2im based on fancy indexing and np.add.at """
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.get_im2col_indices(x_shape, field_height, field_width, padding, stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]

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
        super(ConvLayerPro, self).initialize()
