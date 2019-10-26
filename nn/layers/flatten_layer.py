from .layer import Layer

class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)
        self.input_shape = None

    def forward(self, data):
        # TODO reshape the data here and return it (this can be in place).
        # Input shape: batch x channel x height x width
        # Output shape: batch x (channel x height x width)
        self.input_shape = data.shape
        return data.reshape(data.shape[0], -1)

    def backward(self, previous_partial_gradient):
        # TODO
        return previous_partial_gradient.reshape(self.input_shape)
