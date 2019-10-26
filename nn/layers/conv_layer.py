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

    # @staticmethod
    # @njit(parallel=True, cache=True)
    # def im2col(data, filter_height, filter_width, stride):
    #     # Single image im2col
    #     input_channels, height, width = data.shape
    #     new_height = int((height-filter_height) // stride) + 1
    #     new_width = int((width-filter_width) // stride) + 1
    #     col = np.zeros((new_height*new_width, input_channels*filter_width*filter_width))
    #     for i in prange(new_height):
    #        for j in prange(new_width):
    #            start_h = i*stride
    #            end_h = start_h+filter_height
    #            start_w = j*stride
    #            end_w = start_w+filter_width
    #            patch = data[:, start_h:end_h, start_w:end_w]
    #            col[i*new_width+j, :] = patch.flatten()
    #     return col

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(padded_data, weight, bias, unpadded_shape, padding, stride):
        nimages, input_channels, height, width = unpadded_shape
        nimages, input_channels, padded_height, padded_width = padded_data.shape
        input_channels, output_channels, filter_height, filter_width = weight.shape
        output_height = int((height-filter_height+2*padding) // stride) + 1
        output_width = int((width-filter_width+2*padding) // stride) + 1

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
        padding, p = self.padding, self.padding
        stride = self.stride
        kernel_size = self.kernel_size
        weight = self.weight.data
        bias = self.bias.data
        data_pad = np.pad(data, ((0,0),(0,0),(p,p),(p,p)),
                        'constant',constant_values=(0))

        if (False): # Run serial
            nimages, input_channels, height, width = data.shape
            input_channels, output_channels, filter_height, filter_width = weight.shape
            output_height = int((height-kernel_size+2*padding) // stride) + 1
            output_width = int((width-kernel_size+2*padding) // stride) + 1
            output = np.zeros((nimages, output_channels, output_height, output_width))
            self.colmat_transposed = np.zeros((nimages, output_height*output_width, input_channels*filter_width*filter_width))

        else: # Run numbas
            output, self.colmat_transposed = self.forward_numba(data_pad, weight, bias,
                                        data.shape,  padding, stride)
        self.input_shape = data.shape
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, colmat, original_shape, kernel_shape, kernel_col):
        nimages, output_channels, output_height, output_width = previous_grad.shape
        rubbish, input_channels, height, width = original_shape
        input_channels, output_channels, filter_height, filter_width = kernel_shape

        dkernel = np.zeros((input_channels*filter_height*filter_width, output_channels))
        dinput = np.zeros((nimages, output_height*output_width, input_channels*filter_height*filter_width))
        db = np.zeros((output_channels))
        for n in prange(nimages):
            for c in prange(output_channels):
                image_col = colmat[n, :, :]
                A = previous_grad[n, c, :, :].flatten()
                db[c] += np.mean(A)
                for i in prange(input_channels*filter_height*filter_width):
                    for k in prange(output_height*output_width):
                        dkernel[i, c] += np.multiply(image_col[i,k], A[k])
                        dinput[n, k, i] += np.multiply(A[k], kernel_col[i, c])
        return dinput, dkernel, db

    @staticmethod
    @njit(cache=True, parallel=True)
    def col2im_numba(colmat, original_shape, kernel_shape, output_shape, stride):
        # Col2Im
        nimages, output_channels, output_height, output_width = output_shape
        nimages, input_channels, height, width = original_shape
        input_channels, output_channels, filter_height, filter_width = kernel_shape
        out = np.zeros((nimages, input_channels, height, width))
        for n in prange(nimages):
            for i in prange(output_height):
               for j in prange(output_width):
                   start_h = i*stride
                   end_h = start_h+filter_height
                   start_w = j*stride
                   end_w = start_w+filter_width
                   out[n, :, start_h:end_h, start_w:end_w] = colmat[n, i*output_width+j, :].reshape(input_channels, filter_height, filter_width)
        return out

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba_mixed(previous_grad, colmat_transposed, original_shape, kernel, padding, stride):
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

        if (True): # Backprop
            if (False):
                colmat_transposed = self.colmat_transposed
                nimages, output_channels, output_height, output_width = previous_partial_gradient.shape
                rubbish, input_channels, height, width = original_shape
                input_channels, output_channels, filter_height, filter_width = kernel_shape

                dinput = np.zeros((nimages, input_channels, height+2*padding, width+2*padding))
                dkernel = np.zeros((input_channels*filter_height*filter_width, output_channels))
                db = np.zeros((output_channels))
                for n in prange(nimages):
                    for c in prange(output_channels):
                        A  = np.dot(colmat_transposed[n, :, :], previous_partial_gradient[n,c,:,:].reshape(-1,1))
                        dkernel[:, c] += A.flatten()
                        for i in prange(output_height):
                           for j in prange(output_width):
                               start_h = i*stride
                               end_h = start_h+filter_height
                               start_w = j*stride
                               end_w = start_w+filter_width
                               dinput[n, :, start_h:end_h, start_w:end_w]+=previous_partial_gradient[n,c,i,j]*self.weight.data[:,c,:,:]
                               db[c] += previous_partial_gradient[n,c,i,j]
            else: # Run numba
                dinput, dkernel, db = self.backward_numba_mixed(
                    previous_partial_gradient, self.colmat_transposed,
                     original_shape, self.weight.data, padding, stride)
            dkernel = dkernel.reshape(kernel_shape[0], kernel_shape[2], kernel_shape[3], kernel_shape[1])
            dkernel = np.moveaxis(dkernel, -1, 1)
        else: # Untested backprop
            # Since numba cannot move axis :-(
            kernel_col = self.weight.data
            kernel_col = np.moveaxis(kernel_col, 1, -1)
            kernel_col_moveaxis_shape = kernel_col.shape
            kernel_col = kernel_col.reshape(-1, kernel_col.shape[-1])
            colmat = np.moveaxis(self.colmat_transposed, 1, -1)
            dinput, dkernel, db = self.backward_numba(previous_partial_gradient, colmat,
             original_shape, kernel_shape, kernel_col)


            if (False): # Col2Im
                if (True): # Run without numb
                    nimages, input_channels, height, width = self.input_shape
                    filter_height, filter_width = kernel_size, kernel_size
                    nimages, output_channels, output_height, output_width = previous_partial_gradient.shape
                    out = np.zeros((nimages, input_channels, height, width))
                    out = np.pad(out, ((0,0),(0,0),(p,p),(p,p)),
                                    'constant',constant_values=(0))

                    for n in range(nimages):
                        for i in range(output_height):
                           for j in range(output_width):
                               start_h = i*stride
                               end_h = start_h+filter_height
                               start_w = j*stride
                               end_w = start_w+filter_width
                               out[n, :, start_h:end_h, start_w:end_w] = dinput[n, i*output_width+j, :].reshape(input_channels, filter_height, filter_width)
                else:
                    output_shape = previous_partial_gradient.shape
                    nimages, input_channels, height, width = original_shape
                    padded_original_shape = (nimages, input_channels, height+2*padding, width+2*padding)
                    out = self.col2im_numba(dinput, padded_original_shape, kernel_shape, output_shape, stride)
            kernel_col = kernel_col.reshape(kernel_col_moveaxis_shape)
            kernel_col = np.moveaxis(kernel_col, -1, 1)

        # Update gradient
        self.weight.grad = dkernel
        self.bias.grad = db

        if (p>0):
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
