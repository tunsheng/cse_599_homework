import numpy as np
from numba import njit, prange

from .layer import Layer

class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)
        self.grad = 0
        self.relu = np.vectorize(lambda x: x * (x>0))
        self.drelu = np.vectorize(lambda x: 1.0 * (x>0))
        self.input = 0

    def forward(self, data):
        # TODO
        # Element-wise function
        # Return f(in) shape = (n x d)
        self.input = data
        return self.relu(data)

    def backward(self, previous_partial_gradient):
        # TODO
        if (False):
            print("RELU Layer")
            print("Prev grad = ", previous_partial_gradient.shape)
            print("Grad Weight = ", np.transpose(self.grad).shape)
            print("\n")
        #  dy     dy        df(z)
        # ---- = ------- X -------
        #  dz     df(z)     dz
        # new_grad = dy/df(in) * df(in)/d(in)
        return previous_partial_gradient*(self.input >=0)

    def selfstr(self):
        return str(self.grad.shape)

class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # 3.2) TODO
        shape = data.shape
        output = np.copy(data)
        output = output.flatten()
        for i in prange(len(output)):
            if (output[i]<0):
                output[i]=0
        output = output.reshape(shape)
        output = output.reshape(data.shape)
        return output

    def forward(self, data):
        # Modify if you want
        self.data = data
        output = self.forward_numba(data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # 3.2) # TODO Helper function for computing ReLU gradients
        shape = data.shape
        output = np.zeros(shape)
        data = data.flatten()
        output = output.flatten()
        grad = grad.flatten()
        for i in prange(len(output)):
            if (data[i]>0):
                output[i]=grad[i]
        output = output.reshape(shape)
        data = data.reshape(shape)
        grad = grad.reshape(shape)
        return output

    def backward(self, previous_partial_gradient):
        # TODO
        output=self.backward_numba(self.data, previous_partial_gradient)
        return output
