import numpy as np
import matplotlib.pyplot as plt
import random


# Return function that should be perfomed on x, y coordinate based on operation index
def get_transformation_function(operation):
    if operation == 1:
        get_value = lambda x, y: x * y + 2 * y 

    elif operation == 2:
        get_value = lambda x, y: np.abs(np.cos(x / q) + 2 * np.sin(y / q))

    elif operation == 3:
        get_value = lambda x, y: np.abs(3 * (x / q) - np.cbrt(y / q))

    elif operation == 4: 
        get_value = lambda x, y: random.random()

    return get_value


# Generate image based on user input
def generate_custom_image(operation):
    # Create image matrix and initialize random number generator with given seed
    generated_image = np.zeros((scene_size, scene_size), float)
    random.seed(random_seed)

    # No randomwalk function, get mathematic operation to perform
    if operation != 5:
        operation_function = get_transformation_function(operation)
        for x in range(scene_size):
            for y in range(scene_size):
                generated_image[x, y] = operation_function(x, y)

    else:
        # Randomwalk function
        # Perform randomwalk operation
        x, y = 0, 0
        generated_image[x, y] = 1
        for _ in range(1 + scene_size * scene_size):
            # Find walking direcion
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            x = (x + dx) % scene_size
            y = (y + dy) % scene_size

            # Update corresponding cell
            generated_image[x, y] = 1

    return generated_image


# Normalize image values to 16 bits format
def normalize_image(image):
    max_value = np.max(image)
    min_value = np.min(image)
    image = (2 ** 16 - 1) * ((image - min_value) / (max_value - min_value))
    return image.astype(np.uint16)


# Quantize image to reduce matrix size
def quantize_image(image):
    quantized_image = np.zeros((image_size, image_size), np.uint8)
    step_size = int(np.floor(scene_size / image_size))
    correction_factor = (2 ** 8 - 1) / (2 ** 16 - 1)  # Correction factor to transform from 16 to 8 bits
    for x in range(image_size):
        for y in range(image_size):
            # Update matrix value and perform bit shift for amplitude quantization
            quantized_image[x, y] = correction_factor * image[x * step_size, y * step_size]
            quantized_image[x, y] = quantized_image[x, y] >> (8 - bits_per_pixel)

    return quantized_image


def root_squared_error(image1, image2):
    # Calculate root squared error between two images
    error = 0.0
    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            error += (int(image1[x, y]) - int(image2[x, y])) ** 2

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
    generated_image = generate_custom_image(function)

    # Normalize and quantize image size and amplitude
    normalized_image = normalize_image(generated_image)
    quantized_image = quantize_image(normalized_image)

    # Open original image
    original_image = np.load(filename)

    difference_image = quantized_image - original_image
    print(round(root_squared_error(quantized_image, original_image), 4))
