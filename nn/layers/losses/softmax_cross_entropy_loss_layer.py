import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)
        self.grad = 0

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: N-Dimensional non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets.
            Example: inputs might be (4 x 10), targets (4) and axis 1.
        :param targets: (N-1)-Dimensional class id integers.
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        # TODO

        # Save original shape
        logits_shape_before = logits.shape
        targets_shape = targets.shape

        if axis != -1:
            # Move `axis` dimension to the end
            logits = np.moveaxis(logits, axis, -1)

        logits_shape_moveaxis = logits.shape

        # Make logits and labels into matrices
        logits = self.flatten_outer_dims(logits)
        logits_shape_flatten = logits.shape
        targets = targets.flatten()
        labels = self.one_hot_encode(targets, logits_shape_flatten)

        # Start actual computation
        b = np.max(logits, axis=-1)
        loss=np.zeros([logits_shape_flatten[0], 1]) # n x 1
        grad=np.zeros(logits_shape_flatten) # n x d
        for i in range(logits_shape_flatten[0]):
            summation = np.sum(np.exp(logits[i,:]-b[i]))
            loglikelihood = (logits[i,:]-b[i])-np.log(summation)
            loss[i] = -np.dot(loglikelihood, labels[i,:])

            grad[i, :] = np.exp(loglikelihood)
            grad[i, targets[i]] -= 1

        if (self.reduction=="mean"):# Sum across batch
            batchLoss=np.mean(loss)
            self.grad = grad/logits_shape_flatten[0]
        else:
            batchLoss = np.sum(loss)
            self.grad = grad

        if axis != -1: # Deflat flatten matrix
            logits = self.deflat_dims(logits, logits_shape_moveaxis)
            self.grad = self.deflat_dims(self.grad, logits_shape_moveaxis)

            # Move `axis` dimension to the end
            logits = np.moveaxis(logits, -1, axis)
            self.grad = np.moveaxis(self.grad, -1, axis)


        return batchLoss

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # TODO
        return self.grad

    def flatten_outer_dims(self, tensor):
        # Converts (500,100,200) to (50000,200)
        return tensor.reshape(-1, tensor.shape[-1])

    def deflat_dims(self, tensor, shape):
        return tensor.reshape(shape)

    def one_hot_encode(self, indices, shape):
        """
            Convert indices to 1-hot vector
            indices: 1-d array of indices
            shape:   [len of indices, num_classes]
        """
        output = np.zeros(shape)
        output[np.arange(len(indices)), indices] = 1
        return output
