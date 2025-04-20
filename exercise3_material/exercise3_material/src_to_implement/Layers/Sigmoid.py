import numpy as np
from Layers import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        self.activation_fn = 1 / (1 + np.exp(-input_tensor))
        return self.activation_fn   

    def backward(self, error_tensor):
        sigmoid_derivative = self.activation_fn * (1 - self.activation_fn)
        backprop_signal = error_tensor * sigmoid_derivative 
        return backprop_signal
