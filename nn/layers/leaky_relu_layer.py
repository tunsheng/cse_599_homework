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
        # Store gradient f'(in) shape =  (n x d)
        self.data = data
        #Return f(in) shape = (n x d)
        return self.relu(data)


    def backward(self, previous_partial_gradient):
        # 5.2) TODO
        # (n x c) x  (c x d) = (n x d)
        #  dy     dy          df(x)
        # ---- = --------- X ---------
        #  dx     df(x)       dx

        grad = np.ones_like(self.data)
        grad[self.data<0]=self.slope
        grad[self.data==0]=-0.5
        return previous_partial_gradient*grad

    # @staticmethod
    # @njit(parallel=True, cache=True)
    # def backward_numba(data, grad, rows, cols, middle):
    #     output=np.zeros([rows*cols])
    #     data = data.flatten()
    #     grad = grad.flatten()
    #     for i in prange(rows):
    #         for k in range(middle):
    #             for j in range(cols):
    #                 output[i*cols+j]+=data[i*middle+k]*grad[k*cols+j]
    #     output = output.reshape([rows,cols])
    #     return output
