import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        original_weights = np.copy(weight_tensor)
        sgd_updated_weights = weight_tensor - self.learning_rate * gradient_tensor

        if self.regularizer:
            regularizer_gradient = self.regularizer.calculate_gradient(original_weights)
            sgd_updated_weights -= self.learning_rate * regularizer_gradient

        return sgd_updated_weights


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.sgdM_velocity = 0.

    def calculate_update(self, weight_tensor, gradient_tensor):
        original_weights = np.copy(weight_tensor)
        velocity = self.learning_rate * gradient_tensor + self.momentum_rate * self.sgdM_velocity
        sgdWm_weight_tensor = weight_tensor - velocity
        self.sgdM_velocity = velocity

        if self.regularizer:
            regularizer_gradient = self.regularizer.calculate_gradient(original_weights)
            sgdWm_weight_tensor -= self.learning_rate * regularizer_gradient
        return sgdWm_weight_tensor


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.past_grad_avg = 0.
        self.squared_grad_avg = 0.
        self.b = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        original_weights = np.asarray(weight_tensor).copy()

        self.past_grad_avg = self.mu * self.past_grad_avg + (1 - self.mu) * gradient_tensor
        self.squared_grad_avg = self.rho * self.squared_grad_avg + (1 - self.rho) * np.power(gradient_tensor, 2)
        bias1 = self.past_grad_avg / (1 - np.power(self.mu, self.b))
        bias2 = self.squared_grad_avg / (1 - np.power(self.rho, self.b))
        self.b += 1

        adam_weight_tensor = weight_tensor - self.learning_rate * (bias1 / (np.sqrt(bias2) + np.finfo(float).eps))

        if self.regularizer:
            regularizer_gradient = self.regularizer.calculate_gradient(original_weights)
            adam_weight_tensor -= self.learning_rate * regularizer_gradient

        return adam_weight_tensor
