from .layer import Layer


class SequentialLayer(Layer):
    def __init__(self, layers, parent=None):
        super(SequentialLayer, self).__init__(parent)
        self.layers = layers
        parent = self.parent
        for ll, layer in enumerate(self.layers):
            layer.parent = parent
            setattr(self, str(ll), layer)
            parent = layer

    def forward(self, data):
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def __getitem__(self, item):
        return self.layers[item]

    def backward(self, partial_gradient):
        for layer in self.layers[::-1]:
            partial_gradient = layer.backward(partial_gradient)
        return partial_gradient
