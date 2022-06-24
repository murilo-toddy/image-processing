import numpy as np
import imageio
import random


# Convert RGB image to grayscale using luminance
def luminance(image):
    grayscale_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return np.clip(grayscale_image.astype(np.uint8), 0, 255)


# Calculate euclidian distance between two arrays
def euclidian_distance(m1, m2):
    return np.sqrt(np.sum(np.square(m1 - m2)))


# Find closest centroid to a given pixel
def get_closest_centroid_index(coordinate, centroids):
    closest_centroid_index = 0
    min_distance = euclidian_distance(coordinate, centroids[0])
    # Calculate the closest centroid and return its index
    for index, centroid in enumerate(centroids):
        distance = euclidian_distance(coordinate, centroid)
        if distance < min_distance:
            closest_centroid_index = index
    return closest_centroid_index


# Apply k-Method in image
def k_method(number_of_clusters, image, option, number_of_iters):
    # Initialize centroids
    initial_centroids = np.sort(random.sample(range(0, image.shape[0] * image.shape[1]), number_of_clusters))
    centroids = []
    for centroid in initial_centroids:
        centroids.append(np.array([centroid // image.shape[1], centroid % image.shape[1]]))
    
    print(centroids)
    pixels_in_centroid = [[]] * number_of_clusters
    for _ in range(number_of_iters):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                coordinate = np.array([x, y])
                # For each pixel, save its closest centroid
                pixels_in_centroid[get_closest_centroid_index(coordinate, centroids)].append(np.array([x, y]))
                

    print(pixels_in_centroid)


# Calculate difference between single-color images
def rmse(image1, image2):
    m, n = image1.shape
    return np.sqrt(np.sum((int(image1) - int(image2)) ** 2) / m / n)


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
    input_image = imageio.imread(input_image_name)
    reference_image = imageio.imread(reference_image_name)

    # Apply k-method
    generated_image = k_method(number_of_clusters, input_image, option, number_of_iters)

    if option < 3:
        print(round(colored_rmse(generated_image, reference_image), 4))
    else:
        print(round(rmse(generated_image, reference_image), 4))
