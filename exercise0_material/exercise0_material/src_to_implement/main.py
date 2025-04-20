import numpy as np
import matplotlib.pyplot as plt
from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

def configure_checker():
    resolution = int(input("Enter Checker resolution: "))
    tile_size = int(input("Enter Checker tile_size: "))

    checker = Checker(resolution, tile_size)
    checker.show()

def configure_circle():
    resolution = int(input("Enter Circle resolution: "))
    radius = int(input("Enter Circle radius: "))
    position = tuple(map(int, input("Enter the Circle coordinate as 'x,y': ").split(',')))

    circle = Circle(resolution, radius, position)
    circle.show()

def configure_spectrum():
    resolution = int(input("Enter the Spectrum resolution: "))
    spectrum = Spectrum(resolution)
    spectrum.show()

def setup_image_generator():
    file_path = r"C:\Users\Asus\Desktop\DL WS25\exercise0_material\exercise0_material\src_to_implement\exercise_data"
    label_file = r"C:\Users\Asus\Desktop\DL WS25\exercise0_material\exercise0_material\src_to_implement\Labels.json"
    batch_size = int(input("Enter batch size: "))
    image_size = list(map(int, input("Enter image size as 'height,width,channel': ").split(',')))
    rotation = input("rotation images? (true/false): ").strip().lower() == 'true'
    mirroring = input("data mirroring? (true/false): ").strip().lower() == 'true'
    shuffle = input("data shuffling? (true/false): ").strip().lower() == 'true'

    generator = ImageGenerator(file_path, label_file, batch_size, image_size, rotation, mirroring, shuffle)
    generator.show()

def main_menu():
    part = input("Choose Exercise (1 for pattern generation, 2 for image generation): ")

    if part == "1":
        sub_part = input("Choose pattern type (1 for Checker, 2 for Circle, 3 for Spectrum): ")

        if sub_part == "1":
            configure_checker()
        elif sub_part == "2":
            configure_circle()
        elif sub_part == "3":
            configure_spectrum()
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    elif part == "2":
        setup_image_generator()
    
    else:
        print("Invalid input. Please choose between Exercise 1 or Exercise 2.")

if __name__ == "__main__":
    main_menu()
