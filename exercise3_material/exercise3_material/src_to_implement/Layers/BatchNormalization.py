import numpy as np
from Layers import Base, Helpers
import copy


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.gamma = np.ones(channels)
        self.beta = np.zeros(channels)
        self._optimizer = None

        # Moving averages for test phase
        self.moving_mean = None
        self.moving_variance = None
        self.momentum = 0.8  # Momentum for moving averages
        self.epsilon = 1e-10  # Small value to avoid division by zero

        # Internal placeholders for forward/backward passes
        self.normalized_input = None
        self.mean = None
        self.variance = None
        self.input_tensor = None

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

    def forward(self, input_tensor):
        self.is_conv = len(input_tensor.shape) == 4  # Determine if input is 4D (image)
        if self.is_conv:
            input_tensor = self.reformat(input_tensor)

        self.input_tensor = input_tensor

        if self.testing_phase:
            # Use moving averages for testing
            mean = self.moving_mean
            variance = self.moving_variance
        else:
            # Compute batch statistics for training
            mean = np.mean(input_tensor, axis=0)
            variance = np.var(input_tensor, axis=0)

            # Update moving averages
            if self.moving_mean is None:
                self.moving_mean = mean
                self.moving_variance = variance
            else:
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
                self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * variance

        self.mean = mean
        self.variance = variance

        # Normalize input
        self.normalized_input = (input_tensor - mean) / np.sqrt(variance + self.epsilon)
        output = self.gamma * self.normalized_input + self.beta

        if self.is_conv:
            output = self.reformat(output, inverse=True)

        return output

    def backward(self, error_tensor):
        if self.is_conv:
            error_tensor = self.reformat(error_tensor)

        # Gradients for gamma and beta
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Update weights and biases using optimizer
        if self._optimizer is not None:
            self.gamma = self._optimizer.weight.calculate_update(self.gamma, self.gradient_weights)
            self.beta = self._optimizer.bias.calculate_update(self.beta, self.gradient_bias)

        # Compute gradients w.r.t. inputs
        gradient_input = Helpers.compute_bn_gradients(
            error_tensor, self.input_tensor, self.gamma, self.mean, self.variance + self.epsilon
        )

        if self.is_conv:
            gradient_input = self.reformat(gradient_input, inverse=True)

        return gradient_input


def reformat(self, input_tensor):
    """
    Reformat the input tensor to be 4D for batch normalization.
    The input is expected to be in a format like (batch_size, channels, height, width).
    If it's 2D or 3D, reshape it accordingly.
    """
    # Check if the tensor is already 4D
    if len(input_tensor.shape) == 4:
        return input_tensor
    
    # Handle 3D input (e.g., (channels, height, width)) by adding a batch dimension
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension, making it (1, channels, height, width)
        return input_tensor

    # Handle 2D input (e.g., (channels,)) by reshaping to a dummy 4D (1, channels, 1, 1)
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Shape becomes (1, channels, 1, 1)
        return input_tensor

    # Raise error if tensor is neither 4D, 3D, nor 2D
    raise ValueError("Expected a tensor with 2, 3, or 4 dimensions for reformatting, but got {}".format(len(input_tensor.shape)))


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)

    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta
