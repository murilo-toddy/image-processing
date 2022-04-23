import imageio
import numpy as np


def find_optimal_treshold(image, initial_treshold):
    ti = initial_treshold
    tj = np.average(image)
    while ti - tj > 0.5:
        g1 = []
        g2 = []
        for x in range(image.shape):
            for y in range(image[0].shape):
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
    generated_image = np.zeros((image.shape, image[0].shape), np.uint8)
    for x in range(image.shape):
        for y in range(image[0].shape):
            generated_image[x, y] = 1 if image[x, y] > treshold else 0
    return generated_image


def matrix_to_array(matrix):
    array = []
    for x in range(matrix.shape):
        for y in range(matrix[0].shape):
            array.append(matrix[x, y])
    return array


def array_to_matrix(array, m, n):
    matrix = np.zeros((m, n), np.uint8)
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


def median_filter(image):
    pass


def rmse(image1, image2):
    error = 0
    for x in range(image1.shape):
        for y in range(image1[0].shape):
            error += (image1[x, y] - image2[x, y]) ** 2
    return np.sqrt(error / image1.shape / image1[0].shape)


if __name__ == "__main__":
    # Read user input
    filename = str(input()).rstrip()
    method = int(input())

    original_image = np.load(filename)

    if method == 1:
        generated_image = limiarization(original_image)
    elif method == 2:
        generated_image = filtering_1d(original_image)
    elif method == 3:
        generated_image = filtering_2d(original_image)
    elif method == 4:
        generated_image = median_filter(original_image)

    print(round(rmse(original_image, generated_image), 4))
