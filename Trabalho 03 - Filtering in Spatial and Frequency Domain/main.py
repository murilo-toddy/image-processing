# Name:   Murilo Fantucci Todão
# NUSP:   11299982
# Course: SCC0251
# Year:   2022
# Title:  Assignment 3 - Filtering in Spatial and Frequency Domain
import numpy as np
import imageio


# Convert uint8 matrix to float format
def convert_uint8_float(matrix):
    float_matrix = np.zeros([matrix.shape[0], matrix.shape[1]], float)

    # Divide each entry by 255 to get a value between 0 and 1
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            float_matrix[x, y] = float(matrix[x, y]) / (2 ** 8 - 1)
    return float_matrix


# Convert float matrix to uint8 format
def convert_float_uint8(matrix):
    int_matrix = np.zeros([matrix.shape[0], matrix.shape[1]], np.uint8)

    # Get max and min values
    max_value = np.max(matrix)
    min_value = np.min(matrix)

    # Perform normalization
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            int_matrix[x, y] = int(255 * (matrix[x, y].real - min_value) / (max_value - min_value))
    return int_matrix


# Calculate difference between two images based on given formula
def rmse(image1, image2):
    error = 0
    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            error += (int(image1[x, y].real) - int(image2[x, y].real)) ** 2
    return np.sqrt(error / image1.shape[0] / image1.shape[1])


if __name__ == "__main__":
    # Read user input and load matrices
    input_image_name = str(input()).rstrip()
    filter_name = str(input()).rstrip()
    reference_image_name = str(input()).rstrip()

    input_image = imageio.imread(input_image_name)
    filter_matrix = imageio.imread(filter_name)
    reference_image = imageio.imread(reference_image_name)

    # Convert filter to float format
    float_filter = convert_uint8_float(filter_matrix)

    # Calculate Fourier spectrum of input image and shift it
    dft = np.fft.fft2(input_image)
    dft_shifted = np.fft.fftshift(dft)

    # Perform matrix multiplication with the filter
    dft_filtered = np.multiply(dft_shifted, float_filter)

    # Perform inverse FFT operation
    dft_unshifted = np.fft.fftshift(dft_filtered)
    processed_image_float = np.fft.ifft2(dft_unshifted)
    processed_image = convert_float_uint8(processed_image_float.real)

    # Calculate difference between processed and reference image
    print(round(rmse(processed_image, reference_image), 4))
