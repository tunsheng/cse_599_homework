from abc import ABC
from typing import List, Tuple

from nn.layers import Layer
from nn.layers.losses import LossLayer


class Network(Layer, ABC):
    def __init__(self, loss_layer: LossLayer):
        super(Network, self).__init__()
        self.loss_layer: LossLayer = loss_layer

    def loss(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def backward(self) -> None:
        gradient = self.loss_layer.backward()

        # Create graph
        frontier = [self.loss_layer]
        graph = {}
        while len(frontier) > 0:
            node = frontier.pop()
            if node.parents is None:
                continue
            for parent in node.parents:
                if parent not in graph:
                    graph[parent] = set()
                graph[parent].add(node)
                frontier.append(parent)

        # Topological sort
        order = []
        frontier = [self.loss_layer]
        while len(frontier) > 0:
            node = frontier.pop()
            order.append(node)
            if node.parents is None:
                continue
            for parent in node.parents:
                graph[parent].remove(node)
                if len(graph[parent]) == 0:
                    frontier.append(parent)

        gradients = {}
        for layer in self.loss_layer.parents:
            gradients[layer] = gradient
        # Ignore loss layer because already computed
        order = order[1:]
        # Send gradients backwards
        for layer in order:
            output_grad = layer.backward(gradients[layer])
            if layer.parents is not None:
                assert isinstance(layer.parent, List) == isinstance(
                    output_grad, Tuple
                ), "Gradients should be a list iff there are multiple parents."
                if not isinstance(output_grad, Tuple):
                    output_grad = (output_grad,)
                for parent, grad in zip(layer.parents, output_grad):
                    if parent in gradients:
                        gradients[parent] = gradients[parent] + grad
                    else:
                        gradients[parent] = grad
