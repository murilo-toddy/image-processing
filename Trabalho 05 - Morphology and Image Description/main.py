import numpy as np
import imageio


# Convert image to grayscale using luminance method
def luminance_weights(image):
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)


# Set image values to be binary
def limiarization(image, treshold):
    # Update each cell to be 1 if bigger than treshold and 0 otherwise
    limiarization_image = np.ones([image.shape[0], image.shape[1]]).astype(np.uint8)
    limiarization_image[image < treshold] = 0
    return limiarization_image


# Perform image erosion
def erode_image(image):
    eroded = np.zeros([image.shape[0], image.shape[1]]).astype(np.uint8)
    for x in range(1, image.shape[0]):
        for y in range(1, image.shape[1]):
            # Verify if all elements are one
            if image[x-1:x+2, y-1:y+2].all():
                eroded[x, y] = 1
    return eroded
    

# Perform image dilation
def dilate_image(image):
    dilated = np.zeros([image.shape[0], image.shape[1]]).astype(np.uint8)
    for x in range(1, image.shape[0]):
        for y in range(1, image.shape[1]):
            # Verify if any element is one
            if image[x-1:x+2, y-1:y+2].any():
                dilated[x, y] = 1
    return dilated


# Perform opening morphological transformation
def closing(image):
    return erode_image(dilate_image(image))


# Perform closing morphological transformation
def opening(image):
    return dilate_image(erode_image(image))


# Create masks 1 and 2 using given specification
def generate_masks(grayscale, limiarized):
    mask1 = grayscale.astype(np.uint8) * (1 - limiarized.astype(np.uint8))
    mask2 = grayscale.astype(np.uint8) * limiarized.astype(np.uint8)
    return mask1, mask2


# Find co-occurence matrix
def create_co_occurence_matrix(image, q):
    image_max = image.max()
    co_occurence_matrix = np.zeros([image_max + 1, image_max + 1])
    dx, dy = q
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            # Calculate difference between seed and Q and update increment respective cell
            co_occurence_matrix[image[x, y], image[x + dx, y + dy]] += 1
    # Return normalized matrix
    return co_occurence_matrix / co_occurence_matrix.sum()


""" Methods to find Halaric Descriptors """
def auto_correlation(matrix, row, col):
    return (matrix * (row * col)).sum()


def contrast(matrix, row, col):
    return ((row - col) ** 2 * matrix).sum()


def dissimilarity(matrix, row, col):
    return (np.abs(row - col) * matrix).sum()


def energy(matrix):
    return (matrix ** 2).sum()


def entropy(matrix):
    matrix = matrix[matrix > 0]
    return -(matrix * np.log(matrix)).sum()


def homogeneity(matrix, row, col):
    return (matrix / (1 + (row - col) ** 2)).sum()


def inverse_difference(matrix, row, col):
    return (matrix / (1 + np.abs(row - col))).sum()


def max_prob(matrix):
    return np.max(matrix)


# Compute all descriptors for a given matrix
def compute_haralick_descriptors(matrix):
    # Store all descriptors
    descriptors = []
    
    # Generate intensity matrices
    row, col = np.ogrid[:matrix.shape[0], :matrix.shape[1]]

    descriptors.append(auto_correlation(matrix, row, col))
    descriptors.append(contrast(matrix, row, col))
    descriptors.append(dissimilarity(matrix, row, col))
    descriptors.append(energy(matrix))
    descriptors.append(entropy(matrix))
    descriptors.append(homogeneity(matrix, row, col))
    descriptors.append(inverse_difference(matrix, row, col))
    descriptors.append(max_prob(matrix))
    return descriptors


# Find image descriptors
def get_descriptors(mask1, mask2, q):
    # Calculate co-occurence matrices
    co_occurence_mask1 = create_co_occurence_matrix(mask1, q)
    co_occurence_mask2 = create_co_occurence_matrix(mask2, q)
    # Calculate descriptors
    haralick_desc_mask1 = compute_haralick_descriptors(co_occurence_mask1)
    haralick_desc_mask2 = compute_haralick_descriptors(co_occurence_mask2)
    # Merge two descriptors into single array
    return np.concatenate((haralick_desc_mask1, haralick_desc_mask2), axis=None)


# Get similarity for each image
def get_ranked_similarities(query, descriptors):
    distances = []
    for image in descriptors:
        distances.append(((query - image) ** 2).sum() / query.shape[0])
    similarities = distances / np.max(distances)
    return similarities


if __name__ == "__main__":
    # Read user input
    index = int(input().rstrip())
    q_value = [int(q) for q in input().rstrip().split(" ")]
    f_param = int(input().rstrip())
    limiarization_treshold = int(input().rstrip())
    dataset_size = int(input().rstrip())

    image_names        = []  # Store all image names
    images             = []  # Store all images from dataset
    gray_images        = []  # Store grayscaled images
    limiarized_images  = []  # Store limiarized images

    transformed_images = []  # Store closed or opened images
    masks1             = []  # Store masks
    masks2             = []
    descriptors        = []  # Store descriptors
    
    for i in range(dataset_size):
        # Open new image
        image_names.append(input().rstrip())
        images.append(imageio.imread(image_names[-1]))

        # Save query image
        if i == index:
            query_image_name = image_names[-1]
            query_image = imageio.imread(query_image_name)

        gray_images.append(luminance_weights(images[-1]))  # Convert to grayscale
        limiarized_images.append(limiarization(gray_images[-1], limiarization_treshold))  # Limiarize image

        # Perform morphological operation
        if f_param == 1:
            transformed_images.append(opening(limiarized_images[-1]))
        else:
            transformed_images.append(closing(limiarized_images[-1]))

        # Calculate masks
        masks = generate_masks(gray_images[-1], transformed_images[-1])
        masks1.append(masks[0])
        masks2.append(masks[1])

        # Calculate all descriptors
        descriptors.append(get_descriptors(masks1[-1], masks2[-1], q_value))

    # Calculate similarities and order images
    similarities = get_ranked_similarities(descriptors[index], descriptors)
    order = [x for _, x in sorted(zip(similarities, image_names))]

    # Print output
    print(f"Query: {query_image_name}")
    print("Ranking:")
    for index, image_name in enumerate(order):
        print(f"({index}) {image_name}")
