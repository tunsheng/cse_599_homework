# Test to make sure that I've implemented backprop correctly

import numpy as np
import torch
import torch.nn.functional as F

from nn import Network
from nn import layers
from nn.layers import losses
from nn.layers.block_layers import ResNetBlock
from tests import utils
from torch.autograd import Variable

class Mynet(Network):
    def __init__(self):
        self.layers = layers.SequentialLayer(
            [
                layers.ConvLayer(1, 6, 5),
                layers.MaxPoolLayer(2, 2),
                layers.ReLULayer(),
                layers.ConvLayer(6, 16, 5),
                layers.MaxPoolLayer(2, 2),
                layers.ReLULayer(),
                layers.FlattenLayer(),
                layers.LinearLayer(16 * 7 * 7, 120),
                layers.ReLULayer(),
                layers.LinearLayer(120, 84),
                layers.ReLULayer(),
                layers.LinearLayer(84, 10),
            ]
        )
        loss_layer = losses.SoftmaxCrossEntropyLossLayer(parent=self.layers)
        super(Mynet, self).__init__(loss_layer)

    def forward(self, data):
        return self.layers(data)

    def loss(self, predictions, labels):
        return self.loss_layer(predictions, labels)

class TorchResNetBlock(torch.nn.Module):
    def __init__(self, shape):
        super(TorchResNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(*shape)
        self.conv2 = torch.nn.Conv2d(*shape)

    def forward(self, data):
        input_data = data
        data = self.conv1(data)
        data = F.relu(data)
        data = self.conv2(data)
        data = data + input_data
        data = F.relu(data)
        return data

class TorchFlattenLayer(torch.nn.Module):
    def forward(self, data):
        return data.view(data.shape[0], -1)

class Total(torch.nn.Module):
    def __init__(self):
        super(Total, self).__init__()

        self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(1, 6, 5, 1, 2),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(6, 16, 5, 1, 2),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.ReLU(),
                TorchFlattenLayer(),
                torch.nn.Linear(16 * 7 * 7, 120),
                torch.nn.ReLU(),
                torch.nn.Linear(120, 84),
                torch.nn.ReLU(),
                torch.nn.Linear(84, 10),
        )

    def forward(self, data):
        return self.layers(data)

    def loss(self, predictions, labels):
        return F.cross_entropy(predictions, labels)


def test_MNISTNetwork():
    np.random.seed(0)
    torch.manual_seed(0)
    data = np.random.random((100, 1, 28, 28)).astype(np.float32) * 10 - 5
    labels = np.random.randint(0, 10, 100).astype(np.int64)

    net = Mynet()
    torch_net = Total()

    utils.assign_conv_layer_weights(net.layers[0], torch_net.layers[0])
    utils.assign_conv_layer_weights(net.layers[3], torch_net.layers[3])
    utils.assign_linear_layer_weights(net.layers[7], torch_net.layers[7])
    utils.assign_linear_layer_weights(net.layers[9], torch_net.layers[9])
    utils.assign_linear_layer_weights(net.layers[11], torch_net.layers[11])


    forward = net(data)

    data0 = utils.from_numpy(data).requires_grad_(True)
    torch_forward = torch_net(data0)

    utils.assert_close(forward, torch_forward)
    print("Passed forward")
    loss = net.loss(forward, labels)
    torch_loss = torch_net.loss(torch_forward, utils.from_numpy(labels))
    utils.assert_close(loss, torch_loss)
    print("Passed loss")

    out_grad = net.backward()
    torch_loss.backward()
    utils.assert_close(out_grad, data0.grad, atol=0.01)
    print("Passed grad")

    tolerance = 1e-4
    print("Checking ", net.layers[11], " gradient")
    utils.check_linear_grad_match(net.layers[11], torch_net.layers[11], tolerance=tolerance)

    print("Checking ", net.layers[9], " gradient")
    utils.check_linear_grad_match(net.layers[9], torch_net.layers[9], tolerance=tolerance)

    print("Checking ", net.layers[7], " gradient")
    utils.check_linear_grad_match(net.layers[7], torch_net.layers[7], tolerance=tolerance)

    print("Checking ", net.layers[3], " gradient")
    utils.check_linear_grad_match(net.layers[3], torch_net.layers[3], tolerance=tolerance)

    print("Checking ", net.layers[0], " gradient")
    utils.check_linear_grad_match(net.layers[0], torch_net.layers[0], tolerance=tolerance)
