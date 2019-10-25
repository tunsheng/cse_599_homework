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
        self.im2col = None
        self.initialize()

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
        m, n_C_prev, pad_H, pad_W = data_pad.shape
        n_C_prev, n_C, kernel_size, kernel_size = weights.shape
        n_H = int((n_H_prev-kernel_size+2*padding)/stride)+1
        n_W = int((n_W_prev-kernel_size+2*padding)/stride)+1

        output = np.zeros((m, n_C, n_H, n_W))
        b = np.zeros((n_C, n_H, n_W)).flatten()
        for i in prange(n_C):
            for j in prange(len(b)):
                b[j] = bias[i]
        b = b.reshape(n_C, n_H, n_W)

        # im2col = np.zeros((m, n_C_prev*kernel_size*kernel_size, n_H*n_W))
        # xw = np.zeros((weights.shape[0], n_H*n_W))
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
                        # im2col[i, c*(kernel_size**2):(c+1)*(kernel_size**2), w*n_H+h] = patch.flatten()
                        xw = np.multiply(patch, weights[:,c,:,:])
                        output[i, c, h, w] = np.sum(xw) + np.sum(b[c,:,:])
            # xw[:,:] = 0
            # for s in prange(weights.shape[0]):
            #     for k in prange(weights.shape[1]):
            #         for t in prange(im2col.shape[-1]):
            #             xw[s,t]+= weights[s, k] * im2col[i, k, t]
            #
            # output[i,:,:,:] = xw.reshape(n_C, n_H, n_W) + kernel_size*b
        return output

    def forward(self, data):
        # Declare variables
        padding, p = self.padding, self.padding
        stride = self.stride
        kernel_size = self.kernel_size
        weights = self.weight.data
        bias = self.bias.data

        # weights = np.moveaxis(weights, 1, -1)
        # weights_shape_moveaxis = weights.shape
        # weights = weights.reshape(-1, weights.shape[-1])
        # weights = weights.T
        data_pad = np.pad(data, ((0,0),(0,0),(p,p),(p,p)),
                        'constant',constant_values=(0))
        output = self.forward_numba(data_pad, data.shape, weights, bias,
                                    kernel_size, stride, padding)
        # output = np.moveaxis(output, 2, -1)
        # Save padded data
        self.input_pad = data_pad
        # self.im2col = im2col
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

        m, n_C_prev, n_H_prev, n_W_prev = prev_shape
        m, n_C_prev, pad_H, pad_W = data_pad.shape
        n_C_prev, n_C, kernel_size, kernel_size = weights.shape
        n_H = int((n_H_prev-kernel_size+2*padding)/stride)+1
        n_W = int((n_W_prev-kernel_size+2*padding)/stride)+1

        output = np.zeros((n_C_prev, n_C, n_H_prev, n_W_prev))
        b = np.zeros((n_C, n_H, n_W)).flatten()
        for i in prange(n_C):
            for j in prange(len(b)):
                b[j] = bias[i]
        b = b.reshape(n_C, n_H, n_W)

        for i in prange(m):
            batch_pad = data_pad[i]
            batch_grad = previous_grad[i]
            for c in prange(n_C):
                for h in prange(n_H):
                    for w in range(n_W):
                        vert_start = h*stride
                        vert_end = vert_start+kernel_size
                        horiz_start = w*stride
                        horiz_end = horiz_start+kernel_size
                        patch = batch_pad[:,vert_start:vert_end,horiz_start:horiz_end]
                        xdy = np.multiply(patch.flatten(), previous_grad[:,c,:,:].flatten())
                        # xw = np.multiply(patch, weights[:,c,:,:])
                        output[:, c, h, w] += xdy
                        # output[i, c, h, w] = np.sum(xw) + np.sum(b[c,:,:])

        return output

    def backward(self, previous_partial_gradient):
        # TODO
        # Previous_partial_gradient has the shape of N x CNew x HNew x WNew
        # number of filters = number of channel

        # Declare variables
        padding, p = self.padding, self.padding
        stride = self.stride
        kernel_size = self.kernel_size
        weights = self.weight.data
        bias = self.bias.data
        output = self.backward_numba(previous_partial_gradient, self.input_pad, weights, weights)
        self.weight.grad = output
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
