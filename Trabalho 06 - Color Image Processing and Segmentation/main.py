import numpy as np
import imageio
import random


# Convert RGB image to grayscale using luminance
def luminance(image):
    grayscale_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return np.clip(grayscale_image.astype(np.uint8), 0, 255)


# Calculate euclidian distance between two arrays
def euclidian_distance(m1, m2):
    return np.abs(m1 - m2)


# Convert image into attributes array
def convert_image_to_att_array(image, option):
    m, n = image.shape[0:2]

    if option == 1:
        # RGB
        return np.reshape(image, [m * n, 3])

    elif option == 3:
        # Luminance
        return np.reshape(luminance(image), [m * n, 1])

    xy = np.zeros((m, n, 2))
    for x in range(m):
        for y in range(n):
            xy[x, y] = np.array(x, y)

    if option == 2:
        # RGB XY
        return np.concatenate((np.reshape(image, [m * n, 3]), xy), axis=1)

    if option == 4:
        # Luminance XY
        return np.concatenate((np.reshape(luminance(image), [m * n, 1]), xy), axis=1)


# Find closest centroid to a given pixel
def get_closest_centroid_index(index, centroids):
    closest_centroid_index = 0
    min_distance = euclidian_distance(index, centroids[0])
    # Calculate the closest centroid and return its index
    for idx, centroid in enumerate(centroids):
        distance = euclidian_distance(index, centroid)
        if distance < min_distance:
            closest_centroid_index = idx
    return closest_centroid_index


def get_new_centroid(closest_centroid, i):
    try:
        middle_index = np.floor(closest_centroid[closest_centroid == i].shape[0] / 2)
    except:
        middle_index = 0
    starting_point = np.where(closest_centroid == i)[0][0]
    return starting_point + middle_index


# Apply k-Method in image
def k_method(number_of_clusters, array, number_of_iters):
    # Initialize centroids
    size, features = array.shape
    centroids = np.sort(random.sample(range(0, size), number_of_clusters))

    closest_centroid = np.zeros(size)
    for _ in range(number_of_iters):
        # Find closest centroid for each pixel
        for i in range(size):
            closest_centroid[i] = get_closest_centroid_index(i, centroids)
            
        # Update centroids
        for i in range(number_of_clusters):
            centroids[i] = get_new_centroid(closest_centroid, i)

    # Update each array value to correspond to closest centroid
    for i in range(len(array)):
        array[i] = array[int(centroids[int(closest_centroid[i])])]

    return array


# Calculate difference between single-color images
def rmse(image1, image2):
    m, n = image1.shape
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
    generated_array = k_method(number_of_clusters, att_space, number_of_iters)

    if option < 3:
        if option == 1:
            features = 3
        else:
            features = 5

        generated_image = np.reshape(generated_array, (m, n, features))
        print(round(colored_rmse(generated_image, reference_image), 4))
    
    else:
        if option == 3:
            features = 1
        else:
            features = 3

        generated_image = np.reshape(generated_array, (m, n, features))
        print(round(rmse(generated_image, reference_image), 4))
