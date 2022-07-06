import numpy as np
import imageio
import random


# Convert RGB image to grayscale using luminance
def luminance(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


# Normalize single channel
def normalize_channel(channel, resolution):
    ch_min = np.min(channel)
    ch_max = np.max(channel)
    return ((2 ** resolution - 1) * (channel - ch_min)) / (ch_max - ch_min)


# Normalize image with n channels
def normalize_image(image, resolution=8):
    for channel in range(image.shape[2]):
        image[:, :, channel] = normalize_channel(image[:, :, channel], resolution)
    return image


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


# Calculate distance between two arrays
def euclidian_distance(cell, centroid):
    euc_sum = 0
    for channel in range(cell.shape[0]):
        euc_sum += (cell[channel] - centroid[channel]) ** 2
    return np.sqrt(euc_sum)


# Return index of cluster with smallest distance
def get_closest_cluster(cell, centroids):
    min_dist = np.Infinity 
    # Find smallest distance and update index
    for i, centroid in enumerate(centroids):
        dist = euclidian_distance(cell, centroid)
        if dist < min_dist:
            min_dist = dist
            index = i
    return index


# Apply k-means algoritm
def k_means(array, n_clusters, n_iters):
    size = array.shape[0]
    # Initialize clusters
    initial_centroids_index = np.sort(random.sample(range(0, size), n_clusters)).astype(int)
    centroids = [array[centroid] for centroid in initial_centroids_index]

    for _ in range(n_iters):
        # Store pixels in given centroid
        pixels_in_centroid = [[] for _ in range(n_clusters)]
        # Accumulate sum of pixel values
        new_centroids = [np.zeros(array.shape[1]) for _ in range(n_clusters)]
        for i in range(size):
            # Find closest centroid for each pixel and update new_centroids
            better_cluster_index = get_closest_cluster(array[i], centroids)
            new_centroids[better_cluster_index] += array[i]
            pixels_in_centroid[better_cluster_index].append(i)

        # Average new centroids
        for i in range(n_clusters):
            new_centroids[i] /= len(pixels_in_centroid[i])

        # Update centroids
        centroids = new_centroids

    # Update image array
    for i in range(size):
        array[i] = centroids[get_closest_cluster(array[i], centroids)]
    
    return array


# Single channel RMSE
def channel_rmse(ch1, ch2):
    m, n = ch1.shape[0:2]
    return np.sqrt(np.sum((ch1 - ch2) ** 2) / m / n)


# RGB RMSE
def colored_rmse(image1, image2):
    error = 0
    for channel in range(3):
        error += channel_rmse(image1[:, :, channel], image2[:, :, channel])
    return error / 3


# Grayscale RMSE
def gray_rmse(image1, image2):
    return channel_rmse(image1[:, :, 0], image2)


if __name__ == "__main__":
    # Read user input
    input_image_name = input().rstrip()
    reference_image_name = input().rstrip()
    option = int(input())
    n_clusters = int(input())
    n_iters = int(input())
    seed = int(input())

    # Open images and convert input image to array format
    random.seed(seed)
    input_image = imageio.imread(input_image_name).astype(np.float64)
    input_image_array = convert_image_to_att_array(input_image, option)
    reference_image = imageio.imread(reference_image_name).astype(np.float64)
    m, n = input_image.shape[0:2]
    
    # Apply k-means method
    generated_array = k_means(input_image_array, n_clusters, n_iters)
    
    if option < 3:
        # RGB image
        if option == 1:
            features = 3
        else:
            features = 5

        # Normalize image and calculate RMSE
        generated_image = normalize_image(np.reshape(generated_array, (m, n, features)))
        print(round(colored_rmse(generated_image, reference_image), 4))

    else:
        # Grayscale image
        if option == 3:
            features = 1
        else:
            features = 3

        # Normalize image and calculate RMSE
        generated_image = normalize_image(np.reshape(generated_array, (m, n, features)))
        print(round(gray_rmse(generated_image, reference_image), 4))
