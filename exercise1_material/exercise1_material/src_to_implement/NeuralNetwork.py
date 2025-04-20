import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer  
        self.loss = []  
        self.layers = []  
        self.data_layer = None  
        self.loss_layer = None  

    def forward(self):
    
        input_tensor, self.label_tensor = copy.deepcopy(self.data_layer.next())
        current_tensor = input_tensor
        for layer in self.layers:
            current_tensor = layer.forward(current_tensor)
        ff_loss = self.loss_layer.forward(current_tensor, copy.deepcopy(self.label_tensor))
        return ff_loss   

    def backward(self):
        
        backward_gradient = self.loss_layer.backward(copy.deepcopy(self.label_tensor))
        for layer in reversed(self.layers):
            backward_gradient = layer.backward(backward_gradient)


    def append_layer(self, layer):
        
        if layer.trainable:  
            layer.optimizer = copy.deepcopy(self.optimizer)  
        self.layers.append(layer)  

    def train(self, iterations):
        
        for _ in range(iterations):
            loss_metric = self.forward()
            self.loss.append(loss_metric)
            self.backward()

    def test(self, input_tensor):
        
        current_output = input_tensor
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
