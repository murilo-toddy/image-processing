# Name:   Murilo Fantucci Todão
# NUSP:   11299982
# Course: SCC0251
# Year:   2022
# Title:  Assignment 4 - Image Restoration
import numpy as np
import imageio
from numpy.fft import fft2, ifft2, fftshift


# Create numpy matrix with uint8 format
def create_matrix(m, n):
    return np.zeros((m, n), np.uint8)


# Create numpy matrix with float format
def create_float_matrix(m, n):
    return np.zeros((m, n), float)


# Pad a matrix with certain amount of cols and rows
def pad_matrix(matrix, nums_to_pad, odd_padding=True):
    if odd_padding:
        return np.pad(matrix, (nums_to_pad, nums_to_pad), "constant", constant_values=0)
    else:
        return np.pad(matrix, (nums_to_pad, nums_to_pad + 1), "constant", constant_values=0)


def convert_float_uint8(matrix):
    # Normalize matrix to be between 0 and 1, scale and convert it to uint8 format
    return (255 * matrix / np.max(matrix)).astype(np.uint8)


# Gaussian filter generator as shown in class
def gaussian_filter(k, sigma):
    """Gaussian filter
    :param k: defines the lateral size of the kernel/filter, default 5
    :param sigma: standard deviation (dispersion) of the Gaussian distribution
    :return matrix with a filter [k x k] to be used in convolution operations
    """
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1 / 2) * (np.square(x) + np.square(y)) / np.square(sigma))
    return filt / np.sum(filt)


def create_gaussian_degraded_image(image, deg_filter_size, sigma):
    # Generate gaussian filter matrix
    filt = gaussian_filter(deg_filter_size, sigma)

    num_to_pad = int(image.shape[0] // 2 - filt.shape[0] // 2)
    filt_pad = pad_matrix(filt, num_to_pad, deg_filter_size % 2)

    # Apply blur to image in frequency domain
    I = fft2(image)
    H = fft2(filt_pad)
    G = np.multiply(I, H)

    # Return filter and degraded image in frequency domain
    return H, G


def apply_constrained_least_squares_filtering(G, H, gamma):
    # Define laplacian operator and pad to fit size
    laplacian = np.matrix([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    num_to_pad = int(G.shape[0] // 2 - laplacian.shape[0] // 2)
    laplacian_pad = pad_matrix(laplacian, num_to_pad)

    # Convert laplacian operator to frequency domain and calculate its power spectrum
    P = fft2(laplacian_pad)
    P_power = np.multiply(P, np.conjugate(P))

    # Calculate H conjugate and its power spectrum
    H_conj = np.conjugate(H)
    H_power = np.multiply(H, H_conj)

    # Find image approximation using given expression and return it
    return np.divide(np.multiply(H_conj, G), (H_power + gamma * P_power + 1e-20))


def constrained_least_squares(image):
    # Read user parameters
    k = int(input())
    sigma = float(input())
    gamma = float(input())

    # Degrade image using gaussian filter
    H, G = create_gaussian_degraded_image(image, k, sigma)
    # Apply method to restore original image in frequency domain
    F_hat = apply_constrained_least_squares_filtering(G, H, gamma)

    # Convert restored image to space domain, clipping it in the process
    return np.clip(ifft2(F_hat).real.astype(int), 0, 255)


# Create a motion point spread function as shown in the lectures
def get_motion_psf(shape, degree_angle: float, num_pixel_dist: int = 20) -> np.ndarray:
    """Essa função retorna uma array representando a PSF para um dado ângulo em graus

    Parameters:
    -----------
        dim_x: int
            The width of the image.
        dim_y: int
            The height of the image.
        degree_angle: float
            The angle of the motion blur. Should be in degrees. [0, 360)
        num_pixel_dist: int
            The distance of the motion blur. [0, \infinity).
            Remember that the distance is measured in pixels.
            Greater will be more blurry.

    Returns:
    --------
        np.ndarray
            The point-spread array associated with the motion blur.

    """
    psf = np.zeros(shape)
    center = np.array([shape[0] - 1, shape[1] - 1]) // 2
    radians = degree_angle / 180 * np.pi
    phase = np.array([np.cos(radians), np.sin(radians)])
    for i in range(num_pixel_dist):
        offset_x = int(center[0] - np.round_(i * phase[0]))
        offset_y = int(center[1] - np.round_(i * phase[1]))
        psf[offset_x, offset_y] = 1
    psf /= psf.sum()

    return psf


# Calculate convolution between two images using frequency domain
def fft_convolve2d(x, y):
    freq_convolution = np.multiply(fft2(x), fft2(y))
    space_convolution = fftshift(ifft2(freq_convolution).real)
    return np.clip(space_convolution, 1e-18, np.max(space_convolution))


# Apply iterative method to find image coefficients
def apply_richardson_lucy(image, psf, max_iter):
    # Create constant starting guess
    r0 = np.full(shape=image.shape, fill_value=1, dtype="float64")
    psf_flipped = np.transpose(psf)
    # Apply iterative method
    for _ in range(max_iter):
        den = np.clip(fft_convolve2d(r0, psf), 1e-10, 255)
        img_conv = fft_convolve2d(np.divide(image, den), psf_flipped)
        r_new = np.multiply(r0, img_conv)
        r_new = np.clip(r_new, 1e-8, 255)
        r0 = r_new
    return r_new


def richardson_lucy(image):
    # Read parameters from user input
    psf_angle = int(input())
    max_iter = int(input())
    psf = get_motion_psf(image.shape, psf_angle)

    # Apply method to get image approximation
    filtered_image = apply_richardson_lucy(image, psf, max_iter)
    return convert_float_uint8(filtered_image)


# Calculate difference between two images
def rmse(image1, image2):
    m, n = image1.shape
    return np.sqrt(np.sum((image1.real - image2.real) ** 2) / m / n)


if __name__ == "__main__":
    # Read input image name and method
    input_image_name = str(input()).rstrip()
    method = int(input())

    # Open image and select corresponding method
    input_image = imageio.imread(input_image_name)

    if method == 1:
        selected_method = constrained_least_squares

    elif method == 2:
        selected_method = richardson_lucy

    # Apply method and compare images
    filtered_image = selected_method(input_image)
    print(round(rmse(input_image, filtered_image), 4))
