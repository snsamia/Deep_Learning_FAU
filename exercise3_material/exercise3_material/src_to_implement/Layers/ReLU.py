import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        self.layer_output_nxtLinput = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        relu_backprop = error_tensor.copy()
        relu_backprop[self.layer_output_nxtLinput <= 0] = 0
        return relu_backprop
