import numpy as np
from Layers.Base import BaseLayer
import copy

class FullyConnected(BaseLayer):  
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True 
        self.input_size = input_size 
        self.output_size = output_size  
        self.weights = np.random.uniform(0, 1, (input_size +1, output_size))
        self.bias = np.random.uniform(size = (1, output_size))
        self.gradient_weights = None  
        self.cached_input_bp = None 
        self.layer_output_nxtLinput = None  
        self._optimizer = None
        
    def forward(self, input_tensor):
    
        batch_size = input_tensor.shape[0]
        added_bias_term = np.ones((batch_size, 1))
        biased_input = np.hstack((input_tensor, added_bias_term))
        self.cached_input_bp = biased_input
        self.layer_output_nxtLinput = np.dot(biased_input, self.weights)
        return self.layer_output_nxtLinput

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_instance):
        self._optimizer = optimizer_instance
        self._optimizer.weight = copy.deepcopy(optimizer_instance)
        self._optimizer.bias = copy.deepcopy(optimizer_instance)

    def backward(self, error_tensor):
        
        error_tensor_gradient = np.dot(error_tensor, self.weights[:-1].T)
        self.gradient_weights = np.dot(self.cached_input_bp.T, error_tensor)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(
                self.weights, self.gradient_weights
            )

        return error_tensor_gradient
    

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.output_size)
    


