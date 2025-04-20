import numpy as np

class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum():
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.sgdM_velocity = 0.
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        velocity = self.learning_rate * gradient_tensor + self.momentum_rate * self.sgdM_velocity
        weight_tensor = weight_tensor - velocity
        self.sgdM_velocity = velocity
        return weight_tensor

class Adam():
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.past_grad_avg = 0.
        self.squared_grad_avg = 0.
        self.b = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.past_grad_avg  = self.mu * self.past_grad_avg  + (1 - self.mu) * gradient_tensor
        self.squared_grad_avg = self.rho * self.squared_grad_avg + (1 - self.rho) * np.power(gradient_tensor, 2)
        bias1 = self.past_grad_avg  / (1 - np.power(self.mu, self.b))
        bias2 = self.squared_grad_avg / (1 - np.power(self.rho, self.b))
        self.b += 1
        weight_tensor = weight_tensor - self.learning_rate * (bias1 / (np.sqrt(bias2) + np.finfo(float).eps))
        return weight_tensor
