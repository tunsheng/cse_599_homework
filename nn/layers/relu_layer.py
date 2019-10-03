import numpy as np
from numba import njit, prange

from .layer import Layer


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)
        self.grad = 0

    def forward(self, data):
        # TODO
        #Return f(in) shape = (n x d)

        relu = np.vectorize(lambda x: x if (x>0) else 0)
        drelu = np.vectorize(lambda x: 1 if (x>0) else 0)
        self.grad = drelu(data)
        return relu(data)

    def backward(self, previous_partial_gradient):
        # TODO
        return np.matmul(previous_partial_gradient, np.transpose(self.grad))

    # def selfstr(self):
    #     return str(self.grad.shape)

class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # 3.2) TODO
        output = np.copy(data)
        output = output.flatten()
        relu = np.vectorize(lambda x: x if (x>0) else 0)
        drelu = np.vectorize(lambda x: 1 if (x>0) else 0)
        output = relu(output)
        # for i in range(len(output)):
        #     if (output[i] <  0 ):
        #         output[i]=0
        output = output.reshape(data.shape)
        return output

    def forward(self, data):
        # Modify if you want
        output = self.forward_numba(data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # 3.2) TODO
        output = np.zeros(data.shape)
        output = output.flatten()
        drelu = np.vectorize(lambda x: 1 if (x>0) else 0)
        output = drelu(output)
        # for i in range(len(output)):
        #     if (output[i] <  0 ):
        #         output[i]=0
        #     else:
        #         output[i]=1
        output = output.reshape(data.shape)
        grad = output
        return output

    def backward(self, previous_partial_gradient):
        return None
