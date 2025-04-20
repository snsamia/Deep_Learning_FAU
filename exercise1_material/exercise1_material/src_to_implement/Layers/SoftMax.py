import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()  

    def forward(self, input_tensor):
         
        self.cached_logits = input_tensor
        logits_stable = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_logits_stable = np.exp(logits_stable)
        softmax_probabilities = exp_logits_stable / np.sum(exp_logits_stable, axis=1, keepdims=True)
        self.cached_probabilities = softmax_probabilities
        return softmax_probabilities

    def backward(self, error_tensor):

        weighted_error = error_tensor * self.cached_probabilities
        sum_weighted_error = np.sum(weighted_error, axis=1, keepdims=True)
        softmax_backprop = self.cached_probabilities * (error_tensor - sum_weighted_error)
        
        return softmax_backprop

        