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
<<<<<<< HEAD
        # logits: n x d
        # targets:  n -> (0 to d)
        # grad: n x d
        # loss: n
        # batchLoss: 1
        # b: n x 1
        # CrossEntropy, E = - sum^{nclass} [ target_i * log(predict_i) ]
        # dE/d(predict_i) = - [ target_i / predict_i ]


        """
        Computes softmax cross entropy between logits and label.
        A common use is to have logits and labels of shape
        [batch_size, num_classes] but higher dimensions are
        supported with the axis argument specifying the class
        dimension.

        Backprop will happen into both logits and labels.
        To disallow backprop into labels, pass label tensors
        through tf.stop_gradient before feeding it to this
        function.

        labels: Each vector along the class dimension should
                hold a valid probability distribution
        logits: Unscaled log probabilities
        axis:   The class dimension: default -1

        Return: Same shape as labels except that it does not
                have the last dimension of labels
        """
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

        # print(logits.shape)
        # print(logits_shape_before)
=======
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

>>>>>>> 51143ece6f951518363f24ae28a512b026a8c3f1
        return batchLoss

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # 4.2) TODO
        return self.grad
<<<<<<< HEAD

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

    # def log_soft_max(self, x, b=0):
    #     def log_soft_max_helper(x, b=0):
    #         return (x-b)-np.sum(np.exp(x-b))
    #     output = np.vectorize(log_soft_max_helper)
    #     return log_soft_max_helper
=======
>>>>>>> 51143ece6f951518363f24ae28a512b026a8c3f1
