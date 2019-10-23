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
        self.input = None
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

    def im2col(self, matrix, kernel_size, stride, padding):
        # Input: sptial block of image
        # Output: columns of matrix
        # Assuming filter is a square
        #  In (batch, input_channels, height, width)
        #  Out (input_channels*kernel*kernel, output_height*output_width*batch)
        p=padding
        original_shape = matrix.shape
        paddedMatrix = np.pad(matrix, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        nbatch, nchannels, height, width = np.shape(paddedMatrix)
        outputHeight = int((height-kernel_size)/stride+1)
        outputWidth = int((width-kernel_size)/stride+1)
        output = np.zeros([ nchannels*(kernel_size**2), nbatch*outputWidth*outputHeight ])

        # Different center for different kernel size
        if (kernel_size%2==0):
            offsetLower = 0
            offsetUpper = kernel_size
        else:
            offsetLower = kernel_size//2
            offsetUpper = kernel_size//2 +1

        for k in range(nbatch):
            counter = 0
            batch = k*outputWidth*outputHeight
            for i in range(0, height, stride):
                for j in range(0, width, stride):
                    V = np.pad(np.ones([original_shape[2], original_shape[3]]), ((p,p),(p,p)), mode='constant')
                    if ((p-1)<j) and (j<(width-p)) and ((p-1)<i) and (i<(height-p)):
                        if ((j+offsetUpper) <= width) and ((i+offsetUpper) <= height):
                            patch = paddedMatrix[batch, 0, i-offsetLower:i+offsetUpper, j-offsetLower:j+offsetUpper].reshape(-1,1)
                            for c in range(1, nchannels):
                                patch = np.vstack((patch, paddedMatrix[batch, c, i-offsetLower:i+offsetUpper, j-offsetLower:j+offsetUpper].reshape(-1,1)))
                            output[:, batch+counter] = patch.flatten()
                            counter+= 1
                            # For visualization (debug)
                            if (False):
                                V[ i-offsetLower:i+offsetUpper, j-offsetLower:j+offsetUpper]=2
                                V[i,j]=3
                                plt.axhline(i)
                                plt.axvline(j)
                                plt.imshow(V)
                                plt.pause(0.1)
        return output

    def col2im(self, matrix, original_shape, kernel_size, stride, padding):
        nbatch, nchannels, height, width = original_shape
        paddedHeight = height+2*padding
        paddedWidth = width+2*padding
        paddedMatrix = np.zeros([nbatch*nchannels*paddedHeight*paddedWidth])
        indices = np.array([ i for i in range(np.cumprod(original_shape)[-1])]).reshape(original_shape)
        col_indices = im2col(indices, kernel_size, stride, padding).flatten()
        matrix=matrix.flatten()
        for i in range(len(col_indices)):
            paddedMatrix[int(col_indices[i])] = matrix[i]
        paddedMatrix = paddedMatrix.reshape([nbatch, nchannels, paddedHeight, paddedWidth])
        if padding == 0:
            return paddedMatrix
        return paddedMatrix[:, :, padding:-padding, padding:-padding]

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, weights, bias):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K

        weights = np.moveaxis(weights, 1, -1)
        weights_shape_moveaxis = weights.shape
        weights = weights.reshape(-1, weights.shape[-1]) # (K K  COld) x CNew

        colim = self.im2col()
        output = np.matmul(weights.T, colim)+ bias # CNew x ( N x H x W)
        output = self.col2im()

        # Return to original shape
        weights = weights.reshape(logits_shape_moveaxis)
        weights = np.moveaxis(weights, -1, 1)
        return output

    def forward(self, data):
        # TODO
        self.input = data
        output = forward_numba(self.input, self.weights.data, self.bias.data)
        return None

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, kernel, kernel_grad):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        return None

    def backward(self, previous_partial_gradient):
        # TODO
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
