import numpy as np
from scipy import signal
from Layers import Base
from scipy.signal import correlate2d, convolve2d
import copy

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        if type(stride_shape) == int:
            stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        self.conv2d = (len(convolution_shape) == 3)
        self.weights = np.random.uniform(size = (num_kernels, *convolution_shape))
        if self.conv2d:
            self.convolution_shape = convolution_shape
        else:
            self.convolution_shape = (*convolution_shape, 1)
            self.weights = self.weights[:, :, :, np.newaxis]
        self.num_kernels = num_kernels
        self.bias = np.random.uniform(size = (num_kernels,))
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None
        self.convLastShape = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor 
        if input_tensor.ndim == 3:
            input_tensor = input_tensor[:, :, :, np.newaxis]
            
        self.convLastShape = input_tensor.shape
        
        
        conv_zero_paddedIn = np.zeros((input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] + self.convolution_shape[1] - 1, input_tensor.shape[3] + self.convolution_shape[2] - 1))
        p1 = int(self.convolution_shape[1]//2 == self.convolution_shape[1]/2)
        p2 = int(self.convolution_shape[2]//2 == self.convolution_shape[2]/2)
        if self.convolution_shape[1]//2 == 0 and self.convolution_shape[2]//2 == 0:
            conv_zero_paddedIn = input_tensor
        else:
            conv_zero_paddedIn[:, :, (self.convolution_shape[1]//2):-(self.convolution_shape[1]//2)+p1, (self.convolution_shape[2]//2):-(self.convolution_shape[2]//2)+p2] = input_tensor
            
        input_tensor = conv_zero_paddedIn
        self.conv_padded_input = conv_zero_paddedIn.copy()
        
        h_cnn = np.ceil((conv_zero_paddedIn.shape[2] - self.convolution_shape[1] + 1) / self.stride_shape[0])
        v_cnn = np.ceil((conv_zero_paddedIn.shape[3] - self.convolution_shape[2] + 1) / self.stride_shape[1])
            
        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, int(h_cnn), int(v_cnn)))
        self.output_shape = output_tensor.shape
        
        
        for n in range(input_tensor.shape[0]):
            
            for f in range(self.num_kernels):
                    
                    for i in range(int(h_cnn)):
                        
                        for j in range(int(v_cnn)):
                            
                            if ((i * self.stride_shape[0]) + self.convolution_shape[1] <= input_tensor.shape[2]) and ((j * self.stride_shape[1]) + self.convolution_shape[2] <= input_tensor.shape[3]):
                                output_tensor[n, f, i, j] = np.sum(input_tensor[n, :, i*self.stride_shape[0]:i*self.stride_shape[0] + self.convolution_shape[1], j * self.stride_shape[1]:j * self.stride_shape[1] + self.convolution_shape[2]] * self.weights[f, :, :, :])
                                output_tensor[n, f, i, j] += self.bias[f]
                            else:
                                output_tensor[n, f, i, j] = 0
        if not self.conv2d:
            output_tensor = output_tensor.squeeze(axis = 3) 
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)
        
    def backward(self, error_tensor):
        self.conv_bp_error = error_tensor.reshape(self.output_shape)
        if not self.conv2d:
            self.input_tensor = self.input_tensor[:, :, :, np.newaxis]
        
        self.upsamp_error = np.zeros((self.input_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:]))
        return_tensor = np.zeros(self.input_tensor.shape)
        
        self.de_padded = np.zeros((*self.input_tensor.shape[:2], self.input_tensor.shape[2] + self.convolution_shape[1] - 1,
                                   self.input_tensor.shape[3] + self.convolution_shape[2] - 1))
        
        self.gradient_bias = np.zeros(self.num_kernels)
        
        self.gradient_weights = np.zeros(self.weights.shape)

       
        vertical_padding = int(np.floor(self.convolution_shape[2] / 2))  
        horizontal_padding = int(np.floor(self.convolution_shape[1] / 2))

        for batch in range(self.upsamp_error.shape[0]):
            for kernel in range(self.upsamp_error.shape[1]):
                
                self.gradient_bias[kernel] += np.sum(error_tensor[batch, kernel, :])

                for h in range(self.conv_bp_error.shape[2]):
                    for w in range(self.conv_bp_error.shape[3]):
                        
                        self.upsamp_error[batch, kernel, h * self.stride_shape[0], w * self.stride_shape[1]] = self.conv_bp_error[batch, kernel, h, w]  

                for ch in range(self.input_tensor.shape[1]): 
                    return_tensor[batch, ch, :] += convolve2d(self.upsamp_error[batch, kernel, :], self.weights[kernel, ch, :], 'same')  # zero padding

            
            for n in range(self.input_tensor.shape[1]):
                for h in range(self.de_padded.shape[2]):
                    for w in range(self.de_padded.shape[3]):
                        if (h > horizontal_padding - 1) and (h < self.input_tensor.shape[2] + horizontal_padding):
                            if (w > vertical_padding - 1) and (w < self.input_tensor.shape[3] + vertical_padding):
                                self.de_padded[batch, n, h, w] = self.input_tensor[batch, n, h - horizontal_padding, w - vertical_padding]

            for kernel in range(self.num_kernels):
                for c in range(self.input_tensor.shape[1]):
                    
                    self.gradient_weights[kernel, c, :] += correlate2d(self.de_padded[batch, c, :], self.upsamp_error[batch, kernel, :], 'valid')  # valid padding


        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)

        if not self.conv2d:
            return_tensor = return_tensor.squeeze(axis = 3)
        return return_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)