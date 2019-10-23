from typing import Optional, Callable
import numpy as np

from numba import njit, prange

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
        self.input_pad = None
        self.initialize()

    """
    There is one bias per output channel, similar to the LinearLayer
    (batch, input_channels, height, width)

    padding is defined as how many rows to add to ALL 4 sides
    of the input images. e.g. If an input has shape (2, 3, 100, 100)
    and there is padding of 3, the padded input should be
    of shape (2, 3, 106, 106).

    When computing output sizes, you should discard incomplete rows
    if the stride puts it over the edge. e.g. (2, 3, 5, 5) input,
    3 kernels of size 2x2 and stride of 2 should result in an
    output of shape (2, 3, 2, 2). (batch, output_channels, output_height, output_width)

    You can expect sane sizes of things and don't have to explicitly error check
    e.g. The kernel size will never be larger than the input size
    or larger than the stride.
    """

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data_pad, prev_shape, weights, bias, kernel_size, stride, padding):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        """
        Input:
            data   - column transformed images (Cold x K x K) x (HNew x WNew x N)
            weight - Cold x CNew x K x K
        Transformed
            weight - (Cold x K x K) x CNew
        Output:
            output - column transfromed images CNew x (HNew x WNew x N)
        """
        m, n_C_prev, n_H_prev, n_W_prev = prev_shape
        m, n_Cprev, pad_H, pad_W = data_pad.shape
        n_Cprev, n_C, kernel_size, kernel_size = weights.shape
        n_H = int((n_H_prev-kernel_size+2*padding)/stride)+1
        n_W = int((n_W_prev-kernel_size+2*padding)/stride)+1

        output = np.zeros((m, n_C, n_H, n_W))
        b = np.zeros((n_C, n_H, n_W)).flatten()
        for i in prange(n_C):
            for j in prange(len(b)):
                b[j] = bias[i]
        b = b.reshape(n_C, n_H, n_W)

        for i in prange(m):
            batch_pad = data_pad[i]
            for c in prange(n_C):
                for h in prange(n_H):
                    for w in prange(n_W):
                        vert_start = h*stride
                        vert_end = vert_start+kernel_size
                        horiz_start = w*stride
                        horiz_end = horiz_start+kernel_size
                        patch = batch_pad[:,vert_start:vert_end,horiz_start:horiz_end]
                        xw = np.multiply(patch, weights[:,c,:,:])
                        output[i, c, h, w] = np.sum(xw) + np.sum(b[c,:,:])

        return output

    def forward(self, data):
        # Declare variables
        padding, p = self.padding, self.padding
        stride = self.stride
        kernel_size = self.kernel_size
        weights = self.weight.data
        bias = self.bias.data

        data_pad = np.pad(data, ((0,0),(0,0),(p,p),(p,p)),
                        'constant',constant_values=(0))
        output = self.forward_numba(data_pad, data.shape, weights, bias,
                                    kernel_size, stride, padding)

        # Save paaded data
        self.input_pad = data_pad
        return output

    # def forward(self, data):
    #     # TODO
    #     self.input = data
    #
    #     # Declare variables
    #     padding, p = self.padding, self.padding
    #     stride = self.stride
    #     kernel_size = self.kernel_size
    #     weights = self.weight.data
    #     # original_shape = data.shape
    #     # nbatch, nchannels, height, width = original_shape
    #     # nbatch, nchannels, padded_height, padded_width = padded_matrix.shape
    #     # output_height = int((height-kernel_size)/stride+1)
    #     # output_width = int((width-kernel_size)/stride+1)
    #
    #     # Create padded matrix
    #     padded_matrix = np.pad(data, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
    #     column_data = self.im2col(padded_matrix, kernel_size, stride, padding)
    #
    #     ## Matrix multiplication
    #     weights = np.moveaxis(weights, 1, -1)
    #     weights_shape_moveaxis = weights.shape
    #     weights = weights.reshape(-1, weights.shape[-1])
    #     output = np.matmul(weights.T, column_data)+ self.bias.data
    #     weights = weights.reshape(weights_shape_moveaxis)
    #     weights = np.moveaxis(weights, -1, 1)
    #
    #     # Auxiliary array since numba cant create array Zzz
    #     empty_matrix = np.zeros([ nbatch*nchannels*padded_height*padded_width ])
    #
    #     # Convert back to images
    #     indices = self.create_index(data)
    #     indices = np.pad(indices, ((0,0),(0,0),(p,p),(p,p)), mode='constant', constant_values=-1)
    #     col_indices = self.im2col(indices, self.kernel_size, self.stride, self.padding).flatten()
    #
    #     output = self.col2im(output, empty_matrix, col_indices, data.shape, self.kernel_size, self.stride, self.padding)
    #     output = output.reshape([nbatch, nchannels, padded_height, padded_width])
    #
    #     if padding == 0:
    #         return output
    #     return output[:, :, padding:-padding, padding:-padding]

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, kernel, kernel_grad):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        return None

    def backward(self, previous_partial_gradient):
        print("Shape = ", previous_partial_gradient.shape)
        # TODO
        # Previous_partial_gradient has the shape of N x CNew x HNew x WNew

        
        return None

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
