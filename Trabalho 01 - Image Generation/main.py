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


def generate_custom_image(operation):
    generated_image = np.zeros((scene_size, scene_size), float)
    random.seed(random_seed)

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
        for _ in range(1 + scene_size ** 2):
            # Find walking direcion
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            x = (x + dx) % scene_size
            y = (y + dy) % scene_size
            generated_image[x, y] = 1

    return generated_image


def normalize_image(image):
    max_value = np.max(image)
    min_value = np.min(image)
    image = (2 ** 16 - 1) * ((image - min_value) / (max_value - min_value))
    return image.astype(np.uint16)


def quantize_image(image):
    quantized_image = np.zeros((image_size, image_size), np.uint8)
    step_size = int(np.floor(scene_size / image_size))
    correction_factor = (2 ** 8 - 1) / (2 ** 16 - 1)
    for x in range(image_size):
        for y in range(image_size):
            quantized_image[x, y] = correction_factor * image[x * step_size, y * step_size]
            quantized_image[x, y] = quantized_image[x, y] >> (8 - bits_per_pixel)

    return quantized_image


def root_squared_error(image1, image2):
    # Calculate root squared error between two images
    error_image = np.zeros((image_size, image_size))
    error = 0.0
    for row in range(image1.shape[0]):
        for col in range(image1.shape[1]):
            # if image1[row, col] >= 255 or image2[row, col] >= 255:
                # print(row, col, image1[row, col], image2[row, col])
            greater = max(image1[row, col], image2[row, col])
            smaller = min(image1[row, col], image2[row, col])
            error += (greater - smaller) ** 2
            error_image[row, col] = (greater - smaller) ** 2
            if error_image[row, col] != 0:
                print(row, col, error_image[row, col], image1[row, col], image2[row, col])
            
    # plt.imshow(error_image, cmap="gray")
    # plt.show()
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

    normalized_image = normalize_image(generated_image)
    quantized_image = quantize_image(normalized_image)

    plt.subplot(211); plt.imshow(quantized_image, cmap="gray")

    # Open original image
    original_image = np.load(filename)
    plt.subplot(212); plt.imshow(original_image, cmap="gray")

    difference_image = quantized_image - original_image
    # plt.imshow(difference_image, cmap="gray")

    plt.show()
    print(round(root_squared_error(quantized_image, original_image), 4))
