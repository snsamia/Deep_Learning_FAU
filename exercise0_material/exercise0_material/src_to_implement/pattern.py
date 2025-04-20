import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution must be divisible by 2 * tile_size.")
        self.tile_size = tile_size
        self.resolution = resolution
        self.output = None

    def draw(self):

        amount = int(self.tile_size * 2)
        check_board = np.zeros((amount, amount), dtype=int)
        

        check_board[self.tile_size:, :self.tile_size] = 1
        check_board[:self.tile_size, self.tile_size:] = 1
        
        
        num_repeats = int(self.resolution / (self.tile_size * 2))
    
        checkerboard = np.tile(check_board, (num_repeats, num_repeats))
        
        self.output = checkerboard[:self.resolution, :self.resolution]
        return self.output.copy()

    def show(self):
        plt.imshow(self.draw(), cmap="gray")
        plt.show()




class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        
        x = np.arange(0, self.resolution)
        y = np.arange(0, self.resolution)
        p1, p2 = np.meshgrid(x, y)

        x_coordinate, y_coordinate = self.position
        c_distance = np.sqrt((p1 - x_coordinate) ** 2 + (p2 - y_coordinate) ** 2)

        wh_circle = c_distance <= self.radius
        self.output = wh_circle.copy()  

        return wh_circle

    def show(self):
        plt.imshow(self.draw(), cmap="gray")
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        dim = self.resolution
        rgb_color_array = np.zeros((dim, dim, 3))
        
        rgb_color_array[:, :, 0] = np.linspace(0, 1, dim)
        rgb_color_array[:, :, 1] = np.linspace(0, 1, dim).reshape(dim, 1)
        rgb_color_array[:, :, 2] = np.linspace(1, 0, dim)
        
        self.output = rgb_color_array.copy()
        return rgb_color_array

    def show(self):
        plt.imshow(self.draw())
        plt.show()

