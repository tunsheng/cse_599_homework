import pdb
import numpy as np
import torch
from torch import nn

from nn.layers import ConvLayerPro
from nn.layers import AddLayer
from tests import utils

TOLERANCE = 1e-4

def _test_add_forward(input_shape, out_channels, kernel_size, stride):
    np.random.seed(0)
    torch.manual_seed(0)
    in_channels = input_shape[1]
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    original_input = input.copy()
    layerC = ConvLayerPro(in_channels, out_channels, kernel_size, stride)
    layer = AddLayer(layerC)
    input_tuple = (input, input, input)
    output = layer.forward(input_tuple)

    assert((output==sum(input_tuple)).all)
    assert output.shape == input.shape
    assert(len(out_grad)==len(input_tuple))
    utils.assert_close(output, 3*input, atol=TOLERANCE)

def test_add_forward_batch_input_output():
    width = 100
    height = 100
    kernel_size = 3
    stride = 1
    for batch_size in range(1, 5):
        for input_channels in range(1, 5):
            for output_channels in range(1, 5):
                input_shape = (batch_size, input_channels, width, height)
                _test_add_forward(input_shape, output_channels, kernel_size, stride)


def _test_conv_backward(input_shape, out_channels, kernel_size, stride):
    np.random.seed(0)
    torch.manual_seed(0)
    in_channels = input_shape[1]
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    layerC = ConvLayerPro(in_channels, out_channels, kernel_size, stride)
    layer = AddLayer(layerC)

    input_tuple = (input, input, input)
    prev_grad = 2 * np.ones_like(output) / output.size
    grad_tuple = (prev_grad, prev_grad, prev_grad)

    output = layer.forward(input_tuple)
    out_grad = layer.backward(prev_grad)

    assert output.shape == input.shape
    assert((output==sum(input_tuple)).all)
    assert(len(out_grad)==len(input_tuple))
    utils.assert_close(output, 3*input, atol=TOLERANCE)
    utils.assert_close(out_grad, grad_tuple, atol=TOLERANCE)


def test_conv_backward_batch_input_output():
    width = 10
    height = 10
    kernel_size = 3
    stride = 1
    for batch_size in range(1, 5):
        for input_channels in range(1, 5):
            for output_channels in range(1, 5):
                input_shape = (batch_size, input_channels, width, height)
                _test_conv_backward(input_shape, output_channels, kernel_size, stride)
