import numpy as np  # Manually a
from numba import njit, prange

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = 0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope
        self.data = None
        self.relu = np.vectorize(lambda x: x if (x>0) else x*self.slope)
        self.drelu = np.vectorize(lambda x: 1 if (x>0) else self.slope)

    def forward(self, data):
        # 5.2) TODO
        self.data = data # Store for gradient
        output = self.relu(data)
        # output = self.forward_numba(data, self.slope) # Numba version
        return output


    def backward(self, previous_partial_gradient):
        # 5.2) TODO
        grad = np.ones_like(self.data)
        grad[self.data<0]=self.slope
        grad[self.data==0]=-0.5
        output=previous_partial_gradient*grad
        # output=self.backward_numba(self.data, self.slope, previous_partial_gradient) # Numba
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, slope):
        shape = data.shape
        output = np.copy(data)
        output = output.flatten()
        for i in prange(len(output)):
            if (output[i]<0):
                output[i] *= slope
        output = output.reshape(shape)
        # output = output.reshape(data.shape)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, slope, grad):
        shape = data.shape
        output = np.copy(grad)
        data = data.flatten()
        output = output.flatten()
        for i in prange(len(output)):
            if (data[i]<0):
                output[i]*=slope
            elif (data[i]==0):
                output[i]=-0.5
        output = output.reshape(shape)
        data = data.reshape(shape)
        return output
