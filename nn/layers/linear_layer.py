from typing import Optional, Callable

import numpy as np

from nn import Parameter
from .layer import Layer


class LinearLayer(Layer):
    def __init__(self, input_size: int, output_size: int, parent=None):
        super(LinearLayer, self).__init__(parent)
        self.bias = Parameter(np.zeros((1, output_size), dtype=np.float32))
        self.weight = Parameter(np.zeros((input_size, output_size), dtype=np.float32)) # TODO create the weight parameter
        self.input = 0
        self.initialize()

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Linear layer (fully connected) forward pass
        :param data: n X d array (batch x features)
        :return: n X c array (batch x channels)
        """
        # 3.1) TODO do the linear layer
        # (n x d) (d x c) = (n x c)

        self.input = data # Save this for backward
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

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(LinearLayer, self).initialize()
