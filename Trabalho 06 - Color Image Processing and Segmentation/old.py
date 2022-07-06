# Name:   Murilo Fantucci Todão
# NUSP:   11299982
# Course: SCC0251
# Year:   2022
# Title:  Assignment 6 - Color Image Processing and Segmentation
import time
import numpy as np
import imageio
import random

import matplotlib.pyplot as plt


# Convert RGB image to grayscale using luminance
def luminance(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    

def normalize_channel(channel, resolution):
    channel_min = np.min(channel)
    channel_max = np.max(channel)
    return (((2 ** resolution - 1) * (channel - channel_min)) / np.abs(channel_max - channel_min))


def normalize_image(image, resolution=8):
    for channel in range(image.shape[2]):
        image[:, :, channel] = normalize_channel(image[:, :, channel], resolution)
    return image


# Calculate euclidian distance between two arrays
def euclidian_distance(cell, centroid):
    size = cell.shape[0]
    euclidian_sum = 0
    for i in range(size):
        euclidian_sum += pow(cell[i] - centroid[i], 2)
    return np.sqrt(euclidian_sum)


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


def get_closest_centroid(array, centroids, cell):
    distances = np.zeros(centroids.shape, np.float32)
    for i, centroid in enumerate(centroids):
        distances[i] = euclidian_distance(array[cell], array[centroid])
    return np.argmin(distances)


def get_closest_centroid_comp(array, centroids, cell):
    min_distance = np.Infinity
    for i, centroid in enumerate(centroids):
        dist = euclidian_distance(array[cell], array[centroid])
        if dist < min_distance:
            min_dist_index = i
            min_distance = dist
    return min_dist_index


def update_centroid_value(pixels_in_centroid):
    pixels_in_centroid = np.array(pixels_in_centroid)
    return np.sum(pixels_in_centroid, axis=0) / pixels_in_centroid.shape[0]
    

def new_k(n_clusters, array, n_iters):
    # Generate clusters
    size = array.shape[0]
    centroids = np.sort(random.sample(range(0, size), n_clusters)).astype(int)

    closest_centroid = np.zeros(size, int)
    for _ in range(n_iters):
        pixels_in_centroid = [[] for _ in range(n_clusters)]
        for i in range(size):
            closest_centroid[i] = get_closest_centroid_comp(array, centroids, i)
            pixels_in_centroid[closest_centroid[i]].append(array[i])
        
        for i, centroid in enumerate(centroids):
            array[centroid] = update_centroid_value(pixels_in_centroid[i])


    for i in range(size):
        array[i] = array[centroids[closest_centroid[i]]]

    return array


















def get_closest_cluster(array, index, cluster_values):
    min_dist = np.Infinity
    for i, value in enumerate(cluster_values):
        dist = euclidian_distance(array[index], value)
        if dist < min_dist:
            min_dist = dist
            min_dist_index = i
    return min_dist_index


def k_means(array, n_clusters, n_iters):
    t0 = time.time()
    size = array.shape[0]
    centroids = np.sort(random.sample(range(0, size), n_clusters)).astype(int)
    cluster_values = np.array([array[centroid] for centroid in centroids])
    closest_cluster = np.zeros([cluster_values.shape[0], array.shape[1]])
    pixels_in_cluster = np.zeros(cluster_values.shape[0])
    closest_cluster_pixels = np.zeros(size, int)
    for _ in range(n_iters):
        for i in range(size):
            closest = get_closest_cluster(array, i, cluster_values)
            print(closest)
            closest_cluster_pixels[i] = closest
            closest_cluster[closest] += array[i]
            pixels_in_cluster[closest] += 1

        cluster_values = closest_cluster / pixels_in_cluster[:, None]

    for i in range(size):
        array[i] = cluster_values[closest_cluster_pixels[i]]

    t1 = time.time()

    return array


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
    t0 = time.time()
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
    generated_array = k_means(att_space, number_of_clusters, number_of_iters)

    if option < 3:
        if option == 1:
            features = 3
        else:
            features = 5

        generated_image = normalize_image(np.reshape(generated_array, (m, n, features)))
        print(round(colored_rmse(generated_image, reference_image), 4))
        # plt.imshow(generated_image.astype(np.uint8))

    else:
        if option == 3:
            features = 1
        else:
            features = 3

        generated_image = normalize_image(np.reshape(generated_array, (m, n, features)))
        print(round(rmse(generated_image, reference_image), 4))
        # plt.imshow(generated_image.astype(np.uint8), cmap="gray")

    # t1 = time.time()
    # print(f"elapsed {t1 - t0}")
    # plt.show()
