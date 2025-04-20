import numpy as np

class CrossEntropyLoss:
    def __init__(self):
       
        self.epsilon = np.finfo(float).eps  
        self.cached_predictions = None  

    def forward(self, prediction_tensor, label_tensor):
        
        self.cached_predictions = np.clip(prediction_tensor, self.epsilon, 1.0)
        correct_class_probabilities = -np.sum(label_tensor * np.log(self.cached_predictions), axis=1)
        batch_loss = correct_class_probabilities
        return np.sum(batch_loss)

    def backward(self, label_tensor):
        
        entropy_loss = -label_tensor / self.cached_predictions
        return entropy_loss

