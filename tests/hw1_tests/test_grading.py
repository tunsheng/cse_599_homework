import numpy as np

from nn.layers import *
from tests import utils
import torch

TOLERANCE = 1.e-4

def test_LeakyReLU_backward_easy():
    slope = 0.01
    layer = LeakyReLULayer(slope)
    np.random.seed(1)
    data = np.random.random((10, 20)) * 10 - 5
    data[1, 1] = 0
    output = layer.forward(data)
    previous_partial_gradient = np.ones_like(output) * 2
    output_gradient = layer.backward(previous_partial_gradient)
    assert np.all(output_gradient[data > 0] == 2)
    assert np.all(output_gradient[data < 0] == 2 * slope)
    assert np.all((output_gradient[data == 0] == 0) | (output_gradient[data == 0] == 2 * slope))


def test_LeakyReLU_backward():
    slope = 0.01
    layer = LeakyReLULayer(slope)
    np.random.seed(3)
    data = np.random.random((10, 10, 10, 5, 3)) * 10 - 5
    data[1, 1, 2, 3, 1] = 0
    output = layer.forward(data)
    previous_partial_gradient = np.ones_like(output) * 2
    output_gradient = layer.backward(previous_partial_gradient)
    assert np.all(output_gradient[data > 0] == 2)
    assert np.all(output_gradient[data < 0] == 2 * slope)
    assert np.all((output_gradient[data == 0] == 0) | (output_gradient[data == 0] == 2 * slope))


def test_PRELU_backward_easy():
    np.random.seed(1)
    torch.manual_seed(1)
    slope = 0.02
    n_channels = 10
    torch_layer = torch.nn.PReLU(n_channels, slope)
    layer = PReLULayer(n_channels, slope)
    data = np.random.random((10, n_channels)).astype(np.float32) * 10 - 5
    output = layer.forward(data)
    torch_input = utils.from_numpy(data).requires_grad_(True)
    torch_output = torch_layer(torch_input)
    torch_output.sum().backward()
    out_grad = layer.backward(np.ones_like(output))
    assert np.allclose(utils.to_numpy(torch_input.grad), out_grad, atol=TOLERANCE)
    assert np.allclose(utils.to_numpy(torch_layer.weight.grad), layer.slope.grad, atol=TOLERANCE)


def test_PRELU_backward():
    np.random.seed(3)
    torch.manual_seed(3)
    slope = 0.01
    n_channels = 10
    torch_layer = torch.nn.PReLU(n_channels, slope)
    layer = PReLULayer(n_channels, slope)
    data = np.random.random((10, n_channels, 5, 5)).astype(np.float32) * 10 - 5
    output = layer.forward(data)
    torch_input = utils.from_numpy(data).requires_grad_(True)
    torch_output = torch_layer(torch_input)
    torch_output.sum().backward()
    out_grad = layer.backward(np.ones_like(output))
    print(utils.to_numpy(torch_layer.weight.grad))
    print(layer.slope.grad)
    assert np.allclose(utils.to_numpy(torch_input.grad), out_grad, atol=TOLERANCE)
    assert np.allclose(utils.to_numpy(torch_layer.weight.grad), layer.slope.grad, atol=TOLERANCE)


def test_ReLU_backward_easy():
    layer = ReLULayer()
    np.random.seed(1)
    data = np.random.random((10, 20)) * 10 - 5
    data[1, 1] = 0
    output = layer.forward(data)
    previous_partial_gradient = np.ones_like(output) * 2
    output_gradient = layer.backward(previous_partial_gradient)
    assert np.all(output_gradient[data > 0] == 2)
    assert np.all(output_gradient[data <= 0] == 0)


def test_ReLUNumba_backward_numba_easy():
    layer = ReLUNumbaLayer()
    np.random.seed(3)
    data = np.random.random((10, 20)) * 10 - 5
    data[1, 1] = 0
    output = layer.forward(data)
    previous_partial_gradient = np.ones_like(output) * 2
    output_gradient = layer.backward(previous_partial_gradient)
    assert np.all(output_gradient[data > 0] == 2)
    assert np.all(output_gradient[data <= 0] == 0)


def test_ReLU_backward():
    layer = ReLULayer()
    np.random.seed(5)
    data = np.random.random((10, 10, 10, 5, 3)) * 10 - 5
    data[1, 2, 3, 1, 1] = 0
    output = layer.forward(data)
    previous_partial_gradient = np.ones_like(output) * 2
    output_gradient = layer.backward(previous_partial_gradient)
    assert np.all(output_gradient[data > 0] == 2)
    assert np.all(output_gradient[data <= 0] == 0)


def test_ReLUNumba_backward_numba():
    layer = ReLUNumbaLayer()
    np.random.seed(7)
    data = np.random.random((10, 10, 10, 5, 3)) * 10 - 5
    data[1, 2, 3, 1, 1] = 0
    output = layer.forward(data)
    previous_partial_gradient = np.ones_like(output) * 2
    output_gradient = layer.backward(previous_partial_gradient)
    assert np.all(output_gradient[data > 0] == 2)
    assert np.all(output_gradient[data <= 0] == 0)
