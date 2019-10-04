from numba import njit, prange

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = 0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope

    def forward(self, data):
        # 5.2) TODO
        # Store gradient f'(in) shape =  (n x d)
        self.slope.grad(map(lambda x: 1 if (x>0) else self.slope.data, data))
        #Return f(in) shape = (n x d)
        return map(lambda x: x if (x>0) else x*self.slope.data, data)


    def backward(self, previous_partial_gradient):
        # 5.2) TODO
        # (n x c) x  (c x d) = (n x d)
        #  dy     dy          df(x)
        # ---- = --------- X ---------
        #  dx     df(x)       dx
        return np.matmul(previous_partial_gradient, np.transpose(self.slope.grad))
