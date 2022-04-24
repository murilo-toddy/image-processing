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


def find_optimal_treshold(image, initial_treshold):
    ti = initial_treshold
    tj = np.average(image)
    while ti - tj > 0.5:
        g1 = []
        g2 = []
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if image[x, y] > ti:
                    g1.append(image[x, y])
                else:
                    g2.append(image[x, y])

        average_intensity_1 = sum(g1) / len(g1)
        average_intensity_2 = sum(g2) / len(g2)
        tj = ti
        ti = (average_intensity_1 + average_intensity_2) / 2
    return ti


def limiarization(image):
    initial_treshold = int(input())
    treshold = find_optimal_treshold(image, initial_treshold)
    generated_image = create_matrix(image.shape[0], image.shape[1])
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            generated_image[x, y] = 1 if image[x, y] > treshold else 0
    return generated_image


def matrix_to_array(matrix):
    array = []
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            array.append(matrix[x, y])
    return array


def array_to_matrix(array, m, n):
    matrix = create_matrix(m, n)
    for x in range(m):
        for y in range(n):
            matrix[x, y] = array[x + y * n]
    return matrix


def filtering_1d(image):
    n = int(input())
    weights = input().rstrip().split(" ")
    image_array = matrix_to_array(image)
    output_array = []
    out_size = len(image_array)

    index = 0
    for i in range(n):
        output_array[(index + n / 2) % out_size] += weights[i] * image_array[(index + i) % out_size]
    return array_to_matrix(output_array, image.shape, image[0].shape)


def get_weights_matrix_from_user(n):
    weights = np.zeros((n, n), np.uint8)
    for x in range(n):
        row = input().rstrip().split(" ")
        for y in range(n):
            weights[x, y] = row[y]
    return weights


def filtering_2d(image):
    n = int(input())
    weights = get_weights_matrix_from_user(n)
    x_index = 0
    y_index = 0


def create_matrix_zero_padded(image, rows_to_add):
    scaled_image = create_matrix(image.shape[0] + 2 * rows_to_add, image.shape[1] + 2 * rows_to_add)
    for x in range(image.shape[0] + rows_to_add + 1):
        for y in range(image.shape[1] + rows_to_add + 1):
            scaled_image[x, y] = image[x-rows_to_add, y-rows_to_add] \
                if x in range(rows_to_add, image.shape[0] + rows_to_add) \
                and y in range(rows_to_add, image.shape[1] + rows_to_add) \
                else 0
    return scaled_image


def median_filter(image):
    n = int(input())
    rows_to_add = n // 2
    scaled_image = create_matrix_zero_padded(image, rows_to_add)
    generated_image = create_matrix(image.shape[0], image.shape[1])
    for x in range(rows_to_add, image.shape[0] + rows_to_add):
        for y in range(rows_to_add, image.shape[1] + rows_to_add):
            window = scaled_image[x-rows_to_add:x+rows_to_add+1, y-rows_to_add:y+rows_to_add+1]
            vectorized_window = matrix_to_array(window)
            generated_image[x-rows_to_add, y-rows_to_add] = np.median(vectorized_window)

    return generated_image


def rmse(image1, image2):
    error = 0
    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            error += (int(image1[x, y]) - int(image2[x, y])) ** 2
    return np.sqrt(error / image1.shape[0] / image1.shape[1])


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

    print(round(rmse(original_image, generated_image), 4))
    plt.imshow(generated_image, cmap="gray")
    plt.show()
