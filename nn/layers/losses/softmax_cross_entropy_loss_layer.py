import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed
                          and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)
        self.grad = 0

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: ND non-softmaxed outputs.
                       All dimensions (after removing the "axis" dimension)
                       should have the same length as targets
        :param targets:(N-1)D class id integers.
        :param axis:   Dimension over which to run the Softmax and compare labels.
        :return:       single float of the loss.
        """
        # 4.1) TODO
        # :param logtis: n X d array (batch x features)
        # :param targets: n  array (batch )
        dimInput = logits.shape
        # https://deepnotes.io/softmax-crossentropy
        b = np.max(logits, axis=1)  # Max of each row
        logSoftMax = np.zeros(dimInput)
        logLikelihood = np.zeros([dimInput[0]])
        for i in range(dimInput[0]):
            x = logits[i,:] - b[i]
            logSoftMax[i,:] = x - np.log(np.sum(np.exp(x)))
            logLikelihood[i] = -logSoftMax[i, targets[i]]

        # https://pytorch.org/docs/stable/nn.html?highlight=cross%20entropy#torch.nn.CrossEntropyLoss
        if (reduction=='mean'):
            # 'Default mean'
            batchLoss = np.sum(logLikelihood)
        elif (reduction=='sum'):
            batchLoss = np.sum(logLikelihood)/dimInput[0]
        else:
            # Unreduced
            batchLoss = logLikelihood

        # REMINDER: Next is to fix gradient
        grad = np.exp(logSoftMax)
        # for i in range(dimInput[0]):
        #     grad[i, targets[i]] -= 1.0
        grad[np.arange(targets.shape[0]), targets] -= 1.0
        grad /=dimInput[0] # Must have this
        self.grad = grad

        return batchLoss

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # 4.2) TODO
        return self.grad
