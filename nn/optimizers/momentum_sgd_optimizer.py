from .base_optimizer import BaseOptimizer


class MomentumSGDOptimizer(BaseOptimizer):
    def __init__(self, parameters, learning_rate, momentum=0.9, weight_decay=0):
        super(MomentumSGDOptimizer, self).__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.previous_deltas = [0] * len(parameters)

    def step(self):
        counter=0
        for parameter in self.parameters:
            # 5.1) TODO update the parameters
            dparameter = self.momentum*self.previous_deltas[counter]
            dparameter += (parameter.grad+self.weight_decay*parameter.data)
            self.previous_deltas[counter]=dparameter # Save current gradient
            parameter.data -= self.learning_rate*dparameter
            counter+=1
