import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__()
        self.size = size
        self.slope = Parameter(np.full(size, initial_slope))

    def initialize(self, *args, **kwargs):
        # No need to modify
        pass

    def forward(self, data):
        # 5.2) TODO

        # if (len(self.size)>1):
        print("Size = ", self.size)
        print("Shape = ", data.shape)
        print("Slope =", self.slope.data)
            # prelu = data
            # prelu[data<0]*=self.slope.data[data<0]
        # else:
        #     prelu = 0
        return None

        # m,n = data.shape
        # output = np.copy(data)
        # output = output.flatten()
        # gradient = np.copy(output)
        # alpha = np.copy(self.slope.data).flatten()
        # for i in range(m*n):
        #     if (output[i]>0):
        #         gradient[i] = 1
        #     else:
        #         gradient[i] = 0
        #         output[i] *= alpha[i]
        # output = output.reshape(m, n)
        # self.slope.grad(gradient.reshape(m, n))
        # return output

    def backward(self, previous_partial_gradient):
        # 5.2) TODO
        return np.matmul(previous_partial_gradient, np.transpose(self.slope.grad))


    def selfstr(self):
        return str(self.slope.data.shape)
