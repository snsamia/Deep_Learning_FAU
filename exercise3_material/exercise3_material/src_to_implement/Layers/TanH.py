import numpy as np
from Layers import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        self.output_tensor = np.tanh(input_tensor)
        return self.output_tensor
    
    def backward(self, error_tensor):
        
        tanh_derivative = 1 - np.power(self.output_tensor, 2)
        backprop_gradient_tanH = tanh_derivative * error_tensor
        return backprop_gradient_tanH

   
    
   
