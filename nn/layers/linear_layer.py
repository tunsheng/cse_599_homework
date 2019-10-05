import numpy as np

from nn import Parameter
from .layer import Layer


class LinearLayer(Layer):
    def __init__(self, input_size: int, output_size: int, parent=None):
        super(LinearLayer, self).__init__(parent)
        self.bias = Parameter(np.zeros((1, output_size), dtype=np.float32))
        self.weight = Parameter(np.zeros((input_size, output_size), dtype=np.float32)) # TODO create the weight parameter
        self.input = 0

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Linear layer (fully connected) forward pass
        :param data: n X d array (batch x features)
        :return: n X c array (batch x channels)
        """
        # 3.1) TODO do the linear layer
        # (n x d) (d x c) = (n x c)
        if (False):
            print("Input/Output size = ", self.weight.data.shape)
            print("Data size =", np.shape(data))

        # Save this for backward
        self.input = data
        self.bias.zero_grad()
        self.weight.zero_grad()
        return np.matmul(data, self.weight.data)+self.bias.data

    def backward(self, previous_partial_gradient: np.ndarray) -> np.ndarray:
        """
        Does the backwards computation of gradients wrt weights and inputs
        :param previous_partial_gradient: n X c partial gradients wrt future layer
        :return: gradients wrt inputs
        """
        # 3.1) TODO do the backward step
        # PrevGrd = (n x c)
        # dW = (d x c)
        # dB = (1 x c)
        # dX = (n x d)
        if (False):
            print("Linear Layer")
            print("Prev grad = ", previous_partial_gradient.shape)
            print("Weight = ", self.weight.data.shape)
            print("B = ", self.bias.data.shape)
            print("Data = ", self.input.shape)
            print("\n")

        # Need to save gradient for dw and db
        self.bias.grad=np.sum(previous_partial_gradient, axis=0)
        self.weight.grad=np.matmul(self.input.T, previous_partial_gradient)

        # Return dx
        #  dy     dy          d(wx+b)     dy
        # ---- = --------- X --------- = -------- X w^T
        #  dx     d(wx+b)      dx         d(wx+b)
        return np.matmul(previous_partial_gradient, np.transpose(self.weight.data))

    def selfstr(self):
        return str(self.weight.data.shape)
