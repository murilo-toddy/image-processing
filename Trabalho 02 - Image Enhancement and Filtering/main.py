# Name:   Murilo Fantucci Todão
# NUSP:   11299982
# Course: SCC0251
# Year:   2022
# Title:  Assignment 2 - Image Enhancement and Filtering

import imageio
import numpy as np
import matplotlib.pyplot as plt


# Create a numpy array using uint8 format
def create_matrix(m, n):
    return np.zeros((m, n), np.uint8)


# Create a numpy array using float format
def create_float_matrix(m, n):
    return np.zeros((m, n), float)


# Find optimal treshold for Limiarization algorithm based on specified conditions
def find_optimal_treshold(image, initial_treshold):
    ti = initial_treshold
    # Choose arbitrary starting value
    tj = np.average(image)
    while ti - tj > 0.5:
        g1 = []
        g2 = []
        # Split pixels into G1 and G2 groups
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if image[x, y] > ti:
                    g1.append(image[x, y])
                else:
                    g2.append(image[x, y])

        # Find average intensity of each group
        average_intensity_1 = sum(g1) / len(g1)
        average_intensity_2 = sum(g2) / len(g2)
        tj = ti
        ti = (average_intensity_1 + average_intensity_2) / 2
    return ti


# Convert a matrix to array
def matrix_to_array(matrix):
    array = []
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            array.append(matrix[x, y])
    return array


# Convert an array to a matrix of given dimensions
def array_to_matrix(array, m, n):
    matrix = create_matrix(m, n)
    for x in range(m):
        for y in range(n):
            matrix[x, y] = array[x + y * n]
    return matrix


# Read matrix coefficients from user, row by row
def get_weights_matrix_from_user(n):
    weights = np.zeros((n, n), np.uint8)
    for x in range(n):
        row = input().rstrip().split(" ")
        for y in range(n):
            weights[x, y] = row[y]
    return weights


# Create a copy of the original image padded with zeros
def create_matrix_zero_padded(image, rows_to_add):
    scaled_image = create_matrix(image.shape[0] + 2 * rows_to_add, image.shape[1] + 2 * rows_to_add)
    for x in range(image.shape[0] + rows_to_add + 1):
        for y in range(image.shape[1] + rows_to_add + 1):
            # Value equals original image if it's not in a border
            scaled_image[x, y] = image[x-rows_to_add, y-rows_to_add] \
                if x in range(rows_to_add, image.shape[0] + rows_to_add) \
                and y in range(rows_to_add, image.shape[1] + rows_to_add) \
                else 0
    return scaled_image


# Expand a matrix in symmetric format (new element equals its existing neighbor)
def create_symmetric_matrix(image, rows_to_add):
    symmetric_matrix = create_matrix_zero_padded(image, rows_to_add)
    # Update values for created rows
    for row in range(rows_to_add - 1, -1, -1):
        for y in range(row, image.shape[1] + rows_to_add + 1):
            symmetric_matrix[row, y] = symmetric_matrix[row+1, y]
            symmetric_matrix[image.shape[0] - row + 1, y] = symmetric_matrix[image.shape[0] - row, y]

    # Update values for created columns
    for col in range(rows_to_add - 1, -1, -1):
        for x in range(col, image.shape[0] + rows_to_add + 1):
            symmetric_matrix[x, col] = symmetric_matrix[x, col+1]
            symmetric_matrix[x, image.shape[1] - col + 1] = symmetric_matrix[x, image.shape[1] - col]

    return symmetric_matrix


# Apply limiarization algorithm to an image
def limiarization(image):
    initial_treshold = int(input())

    # Find optimal treshold based on specified algorithm
    treshold = find_optimal_treshold(image, initial_treshold)
    limiarization_image = create_matrix(image.shape[0], image.shape[1])
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Update each cell to be 1 if bigger than treshold and 0 otherwise
            limiarization_image[x, y] = 1 if image[x, y] > treshold else 0
    return limiarization_image


# Apply 1D filtering algorithm to an image
def filtering_1d(image):
    n = int(input())
    # Read weights list from user
    weights = input().rstrip().split(" ")

    # Convert matrix to array
    image_array = matrix_to_array(image)
    output_array = []
    out_size = len(image_array)

    # NEEDS FIXING
    index = 0
    # Calculate weighted sum for each element in the list
    for i in range(n):
        output_array[(index + n // 2) % out_size] += weights[i] * image_array[(index + i) % out_size]

    # Return matrix conversion of calculated list
    return array_to_matrix(output_array, image.shape, image[0].shape)


# Apply 2D filtering algorithm to an image
def filtering_2d(image):
    n = int(input())

    # Read weights matrix from user input
    weights = get_weights_matrix_from_user(n)
    rows_to_add = n // 2

    # Create symmetric matrix
    symmetric_matrix = create_symmetric_matrix(image, rows_to_add)
    filtered_2d_image = create_matrix(image.shape[0], image.shape[1])

    # Loop through each element of original image
    for x in range(rows_to_add, image.shape[0] + rows_to_add):
        for y in range(rows_to_add, image.shape[1] + rows_to_add):
            # Loop through neighbors to find weighted sum
            value = 0
            for dx in range(-rows_to_add, rows_to_add + 1):
                for dy in range(-rows_to_add, rows_to_add + 1):
                    value += weights[dx + rows_to_add, dy + rows_to_add] * int(symmetric_matrix[x + dx, y + dy])
            filtered_2d_image[x - rows_to_add, y - rows_to_add] = value

    return filtered_2d_image


# Apply median filter algorithm to an image
def median_filter(image):
    n = int(input())

    # Create a matrix padded with zeros
    rows_to_add = n // 2
    scaled_image = create_matrix_zero_padded(image, rows_to_add)

    median_filtered_image = create_matrix(image.shape[0], image.shape[1])
    for x in range(rows_to_add, image.shape[0] + rows_to_add):
        for y in range(rows_to_add, image.shape[1] + rows_to_add):
            # Get windowed matrix, convert it to array, sort it and get median
            window = scaled_image[x - rows_to_add:x + rows_to_add + 1, y - rows_to_add:y + rows_to_add + 1]
            vectorized_window = matrix_to_array(window)
            vectorized_window.sort()
            median_filtered_image[x - rows_to_add, y - rows_to_add] = vectorized_window[n // 2]

    return median_filtered_image


# Calculate difference between two images based on given formula
def rmse(image1, image2):
    error = 0
    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            error += (int(image1[x, y]) - int(image2[x, y])) ** 2
    return np.sqrt(error / image1.shape[0] / image1.shape[1])


# def post_processing_normalization(image):
#     image = image / (2 ** 8 - 1)
#     max_value = np.max(image)
#     min_value = np.min(image)
#     image = (2 ** 8 - 1) * ((image - min_value) / (max_value - min_value))
#     return image.astype(np.uint8)


if __name__ == "__main__":
    # Read user input
    filename = str(input()).rstrip()
    method = int(input())

    # Open image as numpy array
    original_image = imageio.imread(filename)

    # Call for specific function based on user input
    if method == 1:
        generated_image = limiarization(original_image)
    elif method == 2:
        generated_image = filtering_1d(original_image)
    elif method == 3:
        generated_image = filtering_2d(original_image)
    elif method == 4:
        generated_image = median_filter(original_image)

    # Normalizes matrix
    # generated_image = post_processing_normalization(generated_image)

    # Calculate error and print rounded to 4 decimal places
    print(round(rmse(original_image, generated_image), 4))
    plt.subplot(121); plt.imshow(original_image, cmap="gray")
    plt.subplot(122); plt.imshow(generated_image, cmap="gray")
    plt.show()
