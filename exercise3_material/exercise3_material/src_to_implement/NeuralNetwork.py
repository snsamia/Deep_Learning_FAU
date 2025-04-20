import copy
import numpy as np

class NeuralNetwork(object):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None

        self.weights_initializer = copy.deepcopy(weights_initializer)
        self.bias_initializer = copy.deepcopy(bias_initializer)

    @property
    def phase(self):
        return self.layers[0].testing_phase

    @phase.setter
    def phase(self, ph_new_mode):
        for layer in self.layers:
            layer.testing_phase = ph_new_mode

    def forward(self):
        input_tensor, self.label_tensor = copy.deepcopy(self.data_layer.next())
        total_reg_loss = 0
        current_tensor = input_tensor
        for layer in self.layers:
            current_tensor = layer.forward(current_tensor)
            if self.optimizer.regularizer is not None and getattr(layer, 'trainable', False):
                total_reg_loss += self.optimizer.regularizer.norm(layer.weights)
        ff_loss = self.loss_layer.forward(current_tensor, copy.deepcopy(self.label_tensor))
        return ff_loss + total_reg_loss

    def backward(self):
        backward_gradient = self.loss_layer.backward(copy.deepcopy(self.label_tensor))
        for layer in reversed(self.layers):
            backward_gradient = layer.backward(backward_gradient)

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
        


    def train(self, iterations):
        self.phase = False
        for _ in range(iterations):
            loss_metric = self.forward()
            self.loss.append(loss_metric)
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        current_output = input_tensor
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
    