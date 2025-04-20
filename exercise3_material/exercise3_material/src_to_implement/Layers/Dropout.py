import numpy as np
from Layers import Base

class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        
    def forward(self, input_tensor):
        if not self.testing_phase:
            binary_mask = np.random.uniform(0, 1, size=input_tensor.shape) < self.probability
            self.scaled_mask = binary_mask.astype(np.float32) / self.probability
        else:
            self.scaled_mask = np.ones_like(input_tensor)   
            
        return input_tensor * self.scaled_mask    
    
    
    def backward(self, error_tensor):
        
        gradient_with_dropout = error_tensor * self.scaled_mask
        return gradient_with_dropout
