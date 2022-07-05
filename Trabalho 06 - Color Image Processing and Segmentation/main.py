import numpy as np
import imageio.v2 as imageio
import random

import matplotlib.pyplot as plt
import time


# Convert RGB image to grayscale using luminance
def luminance(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    

# Calculate euclidian distance between two arrays
def euclidian_distance(cell, centroid):
    return np.sqrt(np.sum(np.square(cell - centroid)))


# Convert image into attributes array
def convert_image_to_att_array(image, option):
    m, n = image.shape[0:2]

    if option == 1:
        # RGB
        return np.reshape(image, [m * n, 3])

    elif option == 3:
        # Luminance
        return np.reshape(luminance(image), [m * n, 1])

    x = np.reshape(np.tile(np.reshape(np.arange(m), [m, 1]), (1, n)), [m * n, 1])
    y = np.reshape(np.tile(np.reshape(np.arange(n), [n, 1]), (m, 1)), [m * n, 1])
    xy = np.concatenate((x, y), axis=1)

    if option == 2:
        # RGB XY
        return np.concatenate((np.reshape(image, [m * n, 3]), xy), axis=1)

    if option == 4:
        # Luminance XY
        return np.concatenate((np.reshape(luminance(image), [m * n, 1]), xy), axis=1)


# Find closest centroid to a given pixel
def get_closest_centroid_index(array, index, centroids):
    closest_centroid_index = 0
    min_distance = euclidian_distance(array[index], array[centroids[0]])
    # Calculate the closest centroid and return its index
    for idx, centroid in enumerate(centroids):
        distance = euclidian_distance(array[index], array[centroid])
        if distance < min_distance:
            closest_centroid_index = idx
            min_distance = distance
    return closest_centroid_index


def get_new_centroid(array, closest_centroid, i):
    filtered_centroid = np.zeros([closest_centroid.shape[0], 1])
    filtered_centroid[closest_centroid == i] = 1
    return (np.dot(array, filtered_centroid) / np.sum(filtered_centroid))


# Apply k-Method in image
def k_method(number_of_clusters, array, max_iter):
    # Initialize centroids
    size, _ = array.shape
    centroids = np.sort(random.sample(range(0, size), number_of_clusters)).astype(int)

    closest_centroid = np.zeros(size, int)
    linear_array = np.arange(0, closest_centroid.shape[0])
    for _ in range(max_iter):
        # Find closest centroid for each pixel
        for i in range(size):
            closest_centroid[i] = get_closest_centroid_index(array, i, centroids)
            
        # Update centroids
        for i in range(number_of_clusters):
            centroids[i] = get_new_centroid(linear_array, closest_centroid, i)

    # Update each array value to correspond to closest centroid
    for i in range(len(array)):
        array[i] = array[centroids[closest_centroid[i]]]

    return array








def get_closest_centroid(cell, centroids):
    distances = np.zeros(centroids.shape, np.float32)
    for i, centroid in enumerate(centroids):
        distances[i] = euclidian_distance(cell, centroid)

    return np.where(distances == np.min(distances))[0][0]


def get_new_centroid(linear, closest_centroid, i):
    if i == 0:
        closest_centroid[closest_centroid != 0] = 0
        closest_centroid[closest_centroid == 0] = 1
    closest_centroid[closest_centroid != i] = 0
    print(closest_centroid.sum())
    return np.dot(linear, closest_centroid) / closest_centroid.sum() / i


def new_k(n_clusters, array, n_iters):
    # Generate clusters
    size = array.shape[0]
    centroids = np.sort(random.sample(range(0, size), n_clusters)).astype(int)

    closest_centroid = np.zeros(size, int)
    linear = np.arange(0, closest_centroid.shape[0])
    for _ in range(n_iters):
        for i in range(len(array)):
            closest_centroid[i] = get_closest_centroid(array[i], centroids)

        for i, centroid in enumerate(centroids):
            centroids[i] = get_new_centroid(linear, closest_centroid, i)


# Calculate difference between single-color images
def rmse(image1, image2):
    m, n = image1.shape[0:2]
    return np.sqrt(np.sum((image1 - image2) ** 2) / m / n)


# Calculate difference between colored images
def colored_rmse(image1, image2):
    error = 0
    for i in range(3):
        error += rmse(image1[:, :, i], image2[:, :, i])
    return error / 3


if __name__ == "__main__":
    # Read user input
    input_image_name = input().rstrip()
    reference_image_name = input().rstrip()
    option = int(input())
    number_of_clusters = int(input())
    number_of_iters = int(input())
    seed = int(input())
    random.seed(seed)

    # Open images
    input_image = imageio.imread(input_image_name).astype(np.float32)
    m, n = input_image.shape[0:2]
    att_space = convert_image_to_att_array(input_image, option)

    reference_image = imageio.imread(reference_image_name).astype(np.float32)

    # Apply k-method
    generated_array = new_k(number_of_clusters, att_space, number_of_iters)

    if option < 3:
        if option == 1:
            features = 3
        else:
            features = 5

        generated_image = np.reshape(generated_array, (m, n, features))
        print(round(colored_rmse(generated_image, reference_image), 4))
        plt.imshow(generated_image.astype(np.uint8))

    else:
        if option == 3:
            features = 1
        else:
            features = 3

        generated_image = np.reshape(generated_array, (m, n, features))
        print(round(rmse(generated_image, reference_image), 4))
        plt.imshow(generated_image.astype(np.uint8), cmap="gray")

    plt.show()