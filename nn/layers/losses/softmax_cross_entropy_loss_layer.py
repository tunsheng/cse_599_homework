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
        dimAxis = logits.shape
        print("Logits = ", dimAxis)
        print("targets = ", targets.shape)
        print("Axis = ", axis)
        # https://deepnotes.io/softmax-crossentropy
        b = np.max(logits, axis=axis)  # Max of each row
        # x = np.array([ logits[i,:]-b[i] for i in range(dimAxis[0]) ])
        if (self.reduction=="mean"):
            x = np.mean(logits, axis=axis)
        else:
            x = np.sum(logits, axis=axis)
        print("x = ", x.shape)
        normalization = np.sum(np.exp(x))
        print("Normalization = ", normalization)

        logSoftMax = x - np.log(normalization)
        print("logSoftMax = ", logSoftMax.shape)
        print("target type = ", targets.dtype)
        logLikelihood=-logSoftMax[targets]
        print("logLikelihood = ", logLikelihood.shape)
        loss = np.sum(logLikelihood)/targets.shape[0]
        print("Loss = ", loss)
        # logitsReduced = np.delete(logits, np.arange(0, dimAxis[axis]-1),
        #                             axis=axis)

        # REMINDER: Next is to fix gradient
        grad = np.exp(logSoftMax)
        grad[range(targets.shape[0], targets)] -= 1
        grad /= targets.shape[0]
        self.grad = grad

        return loss

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # 4.2) TODO


        return self.grad
