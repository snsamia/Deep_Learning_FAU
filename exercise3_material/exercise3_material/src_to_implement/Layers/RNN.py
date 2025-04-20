import numpy as np
import copy 
from Layers import Base
from Layers import FullyConnected as DenseLayer
from Layers import Sigmoid as SigmoidActivation
from Layers import TanH as TanhActivation

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.memorize = False
        self._optimizer = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_fc_layer = DenseLayer.FullyConnected(input_size + hidden_size, hidden_size)
        self.hidden_states = [np.zeros(self.hidden_size)]
        self.tanh_activation = TanhActivation.TanH()
        self.sigmoid_activation = SigmoidActivation.Sigmoid()
        self.output_fc_layer = DenseLayer.FullyConnected(hidden_size, output_size)
        self._gradient_weights = None
        self.output_gradients = None
        self._layer_weights = self.hidden_fc_layer.weights

    def initialize(self, weight_initializer, bias_initializer):
        self.hidden_fc_layer.initialize(weight_initializer, bias_initializer)
        self.output_fc_layer.initialize(weight_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.zeros((self.input_tensor.shape[0], self.output_size))
        if not self.memorize:
            self.hidden_states = [np.zeros((1, self.hidden_size))]
        
        for t, current_input in enumerate(input_tensor):
            current_input = current_input[np.newaxis, :]
            previous_hidden = self.hidden_states[-1].flatten()[np.newaxis, :]
            concatenated_input = np.concatenate((current_input, previous_hidden), axis=1)
            self.hidden_states.append(
                self.tanh_activation.forward(self.hidden_fc_layer.forward(concatenated_input))
            )
            self.output_tensor[t] = self.sigmoid_activation.forward(
                self.output_fc_layer.forward(self.hidden_states[-1])
            )
        
        return self.output_tensor

    def backward(self, error_tensor):
        self._gradient_weights = np.zeros(self.hidden_fc_layer.weights.shape)
        self.output_gradients = np.zeros(self.output_fc_layer.weights.shape)
        propagated_errors = np.zeros((self.input_tensor.shape[0], self.input_size))
        hidden_error = np.zeros((1, self.hidden_size))

        for t in reversed(range(error_tensor.shape[0])):
            current_input = self.input_tensor[t][np.newaxis, :]
            previous_hidden = self.hidden_states[t].flatten()[np.newaxis, :]
            concatenated_input = np.concatenate((current_input, previous_hidden), axis=1)
            
            hidden_layer_output = self.hidden_fc_layer.forward(concatenated_input)
            tanh_output = self.tanh_activation.forward(hidden_layer_output)
            output_layer_output = self.output_fc_layer.forward(tanh_output)
            self.sigmoid_activation.forward(output_layer_output)
            
            gradient = self.hidden_fc_layer.backward(
                self.tanh_activation.backward(
                    self.output_fc_layer.backward(
                        self.sigmoid_activation.backward(error_tensor[t, :])
                    ) + hidden_error
                )
            )
            self._gradient_weights += self.hidden_fc_layer.gradient_weights
            self.output_gradients += self.output_fc_layer.gradient_weights
            propagated_errors[t], hidden_error = (
                gradient[:, :self.input_size].copy(),
                gradient[:, self.input_size:].copy()
            )

        if self._optimizer:
            self.hidden_fc_layer.weights = self._optimizer.calculate_update(
                self.hidden_fc_layer.weights, self._gradient_weights
            )
            self.output_fc_layer.weights = self._optimizer.calculate_update(
                self.output_fc_layer.weights, self.output_gradients
            )

        return propagated_errors

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self.hidden_fc_layer.weights

    @weights.setter
    def weights(self, new_weights):
        self.hidden_fc_layer.weights = new_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, new_gradients):
        self.hidden_fc_layer._gradient_weights = new_gradients






        
        
        




