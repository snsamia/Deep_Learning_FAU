import numpy as np
from Layers import Base


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        
        self.input_shape = input_tensor.shape
        batch_size, num_channels, height, width = input_tensor.shape
        filter_h, filter_w = self.pooling_shape
        vertical_stride, horizontal_stride = self.stride_shape

        
        out_height = (height - filter_h) // vertical_stride + 1
        out_width = (width - filter_w) // horizontal_stride + 1

        output_tensor = np.zeros((batch_size, num_channels, out_height, out_width))

        
        self.max_indices_x = np.zeros_like(output_tensor, dtype=int)
        self.max_indices_y = np.zeros_like(output_tensor, dtype=int)

        
        for h in range(out_height):
            for w in range(out_width):
                h_start, h_end = h * vertical_stride, h * vertical_stride + filter_h
                w_start, w_end = w * horizontal_stride, w * horizontal_stride + filter_w

                pooling_region = input_tensor[:, :, h_start:h_end, w_start:w_end]
                reshaped_region = pooling_region.reshape(batch_size, num_channels, -1)

                
                max_indices = np.argmax(reshaped_region, axis=2)
                output_tensor[:, :, h, w] = np.take_along_axis(
                    reshaped_region, max_indices[:, :, None], axis=2
                ).squeeze(axis=2)

                self.max_indices_x[:, :, h, w], self.max_indices_y[:, :, h, w] = divmod(
                    max_indices, filter_w
                )

        return output_tensor

    def backward(self, error_tensor):
        
        return_tensor = np.zeros(self.input_shape)
        batch_size, num_channels, out_height, out_width = error_tensor.shape

        
        for h in range(out_height):
            for w in range(out_width):
                h_start, w_start = h * self.stride_shape[0], w * self.stride_shape[1]
                max_pos_x = self.max_indices_x[:, :, h, w] + h_start
                max_pos_y = self.max_indices_y[:, :, h, w] + w_start

                batch_indices, channel_indices = np.meshgrid(
                    np.arange(batch_size), np.arange(num_channels), indexing="ij"
                )

                return_tensor[batch_indices, channel_indices, max_pos_x, max_pos_y] += error_tensor[
                    batch_indices, channel_indices, h, w
                ]

        return return_tensor
