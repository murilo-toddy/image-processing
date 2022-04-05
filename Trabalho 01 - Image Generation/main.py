import numpy as np
import imageio
import matplotlib.pyplot as plt
import random


def get_transformation_function(operation):
    if operation == 1:
        def get_value(x, y):
            return x * y + 2 * y

    elif operation == 2:
        def get_value(x, y):
            return np.abs(np.cos(x / q) + 2 * np.sin(y / q))

    elif operation == 3:
        def get_value(x, y):
            return np.abs(3 * (x / q) - np.cbrt(y / q))

    elif operation == 4:
        def get_value(x, y):
            random.random()

    return get_value if get_value else None


def generate_custom_image(operation):
    transformed_image = np.zeros((scene_size, scene_size), float)
    random.seed(random_seed)
    print(transformed_image)

    operation_function = get_transformation_function(operation)

    for x in range(0, transformed_image.shape[0]):
        for y in range(0, transformed_image.shape[1]):
            transformed_image[x, y] = operation_function(x, y)

    plt.plot(transformed_image)
    plt.show()
    return transformed_image


def root_squared_error(image1, image2):
    # Calculate root squared error between two images
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

    # Generate image based on input function
    generate_custom_image(function)

    # Open original image
    original_image = np.load(filename)

