import numpy as np
import imageio
import matplotlib.pyplot as plt
import random


def transform_image(operation, q=None):
    transformed_image = np.zeros((scene_size, scene_size), float)
    random.seed(random_seed)
    print(transformed_image)
    if operation == 1:
        for row in range(transformed_image.shape[0]):
            for col in range(transformed_image.shape[1]):
                transformed_image[row, col] = row * col + 2 * col

    elif operation == 2:
        for row in range(transformed_image.shape[0]):
            for col in range(transformed_image.shape[1]):
                transformed_image[row, col] = np.abs(np.cos(row / q) + 2 * np.sin(col / q))

    elif operation == 3:
        for row in range(transformed_image.shape[0]):
            for col in range(transformed_image.shape[1]):
                transformed_image[row, col] = np.abs(3 * row / q - np.cbrt(row / q))

    elif operation == 4:
        for row in range(transformed_image.shape[0]):
            for col in range(transformed_image.shape[1]):
                transformed_image[row, col] = random.random()

    return transformed_image


def root_squared_error(image1, image2):
    error = 0
    for row in range(image1.shape[0]):
        for col in range(image1.shape[1]):
            error += (image2[row, col] - image1[row, col]) ** 2

    return np.sqrt(error)


if __name__ == "__main__":
    # Read user input
    filename = str(input()).rstrip()
    scene_size = int(input())
    function = int(input())
    q = int(input())
    image_size = int(input())
    bits_per_pixel = int(input())
    random_seed = int(input())

    transform_image(function, q)

    original_image = np.load(filename)
