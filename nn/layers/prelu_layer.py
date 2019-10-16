import numpy as np
from numba import njit, prange, vectorize

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__()
        self.size = size
        self.slope = Parameter(np.full(size, initial_slope))
        self.data = None
        self.slope_full = None

    def initialize(self, *args, **kwargs):
        # No need to modify
        pass

    def selfstr(self):
        return str(self.slope.data)

    def forward(self, data):
        # 5.2) TODO

        # Preprocessing
        self.slope_full = self.slope.data
        if len(self.slope_full) == 1:
            self.slope_full = self.slope_full.flatten()
            self.slope_full = np.full(data.shape[1], self.slope_full[0], dtype=float)

        self.data = data
        original_shape = data.shape
        data = np.moveaxis(data, 1, -1)
        moveaxis_shape = data.shape
        data = self.flatten_outer_dims(data) # Flattern outer dims
        flatten_shape = data.shape
        output = np.copy(data)
        temp = np.copy(data)
        # output = self.forward_numba_full(temp, self.slope_full)
        # Go over each channel
        for i in range(flatten_shape[1]):
            output[:,i] = self.forward_numba(temp[:,i], self.slope_full[i])
        # Postprocessing: Deflate data and move axis back to orignal location
        output = self.deflat_dims(output, moveaxis_shape)
        data = self.deflat_dims(data, moveaxis_shape)
        output = np.moveaxis(output, -1, 1)
        data = np.moveaxis(data, -1, 1)
        return output

    def backward(self, previous_partial_gradient):
        # 5.2) TODO
        # Preprocessing
        slope_grad = np.zeros(self.data.shape[1])
        self.data = np.moveaxis(self.data, 1, -1)
        previous_partial_gradient = np.moveaxis(previous_partial_gradient, 1, -1)
        moveaxis_shape = self.data.shape
        self.data = self.flatten_outer_dims(self.data) # Flattern outer dims
        previous_partial_gradient = self.flatten_outer_dims(previous_partial_gradient)
        flatten_shape = self.data.shape
        output = np.zeros(flatten_shape)
        temp = np.copy(previous_partial_gradient)

        # output = self.backward_numba_full(self.data, self.slope_full, temp)
        # Go over each channel
        for i in range(flatten_shape[1]):
            output[:,i] = self.backward_numba(self.data[:,i], self.slope_full[i],
                                                temp[:,i])
            # slope_grad[i] = np.sum((self.data[:,i] < 0)*self.data[:,i])
        df_dalpha = np.zeros(flatten_shape)
        df_dalpha[self.data < 0] = self.data[self.data < 0]

        dL_dalpha = np.matmul(df_dalpha.T, previous_partial_gradient)
        slope_grad = np.sum(dL_dalpha, axis=-1)

        if len(self.slope.data) == 1:
            self.slope.grad[0] = np.mean(slope_grad)
        else:
            for i in range(flatten_shape[1]):
                self.slope.grad[i] = slope_grad[i]

        # Deflate data and move axis back to orignal location
        output = self.deflat_dims(output, moveaxis_shape)
        self.data = self.deflat_dims(self.data, moveaxis_shape)
        previous_partial_gradient = self.deflat_dims(previous_partial_gradient, moveaxis_shape)
        output = np.moveaxis(output, -1, 1)
        self.data = np.moveaxis(self.data, -1, 1)
        previous_partial_gradient = np.moveaxis(previous_partial_gradient, -1, 1)
        return output

    @staticmethod
    # @njit(parallel=False, cache=False)
    def forward_numba(data, slope):
        shape = data.shape
        output = data
        output = output.flatten()
        data = data.flatten()
        for i in range(len(output)):
            if (output[i] < 0):
                output[i] = data[i] * slope
        output = output.reshape(shape)
        data = data.reshape(data.shape)
        return output

    @staticmethod
    # @njit(parallel=False, cache=False)
    def backward_numba(data, slope, grad):
        shape = data.shape
        output = grad
        data = data.flatten()
        output = output.flatten()
        for i in range(len(output)):
            if (data[i] < 0):
                output[i] *= slope
            elif (data[i] == 0):
                output[i] += -0.5
        output = output.reshape(shape)
        data = data.reshape(shape)
        return output

    # @staticmethod
    # @njit(parallel=True, cache=True)
    # def forward_numba_full(data, slope):
    #     output = data
    #     for i in range(data.shape[1]):
    #         for j in prange(data.shape[0]):
    #             if (output[i,j] < 0):
    #                 output[i,j] *= slope[i]
    #     return output
    #
    # @staticmethod
    # @njit(parallel=True, cache=True)
    # def backward_numba_full(data, slope, grad):
    #     output = grad
    #     for i in range(data.shape[1]):
    #         for j in prange(data.shape[0]):
    #             if (data[i,j] < 0):
    #                 output[i,j] *= slope[i]
    #             elif (data[i,j] == 0):
    #                 output[i,j] += -0.5
    #     return output

    def flatten_outer_dims(self, tensor):
        return tensor.reshape(-1, tensor.shape[-1])

    def deflat_dims(self, tensor, shape):
        return tensor.reshape(shape)
