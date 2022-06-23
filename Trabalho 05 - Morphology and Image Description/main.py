import numpy as np
import imageio


# Convert image to grayscale using luminance method
def luminance_weights(image):
    matrix = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(float)
    return np.clip(matrix, 0, 255)


# Set image values to be binary
def limiarization(image, treshold):
    # Update each cell to be 1 if bigger than treshold and 0 otherwise
    limiarization_image = np.where(image >= treshold, 1, 0)
    return limiarization_image.astype(np.uint8)


# Perform image erosion
def erode_image(image):
    # Create 3x3 rectangle filled with ones
    structuring_kernel = np.full(shape=(3, 3), fill_value=1)

    # Pad image to account for borders
    image_pad = np.pad(array=image, pad_width=1, mode="edge")
    eroded = np.zeros([image.shape[0], image.shape[1]])
    for x in range(2, image.shape[0] + 1):
        for y in range(2, image.shape[1] + 1):
            # Generate submatrix and verify if any element is equal to kernel
            sub_matrix = image_pad[x-1:x+2, y-1:y+2]
            eroded[x-2, y-2] = np.array(1 if (sub_matrix == structuring_kernel).all() else 0)

    return eroded
    

# Perform image dilation
def dilate_image(image):
    # Create 3x3 rectangle filled with ones
    structuring_kernel = np.full(shape=(3, 3), fill_value=1)

    # Pad image to account for borders
    image_pad = np.pad(array=image, pad_width=1, mode="edge")
    dilated = np.zeros([image.shape[0], image.shape[1]])
    for x in range(2, image.shape[0] + 1):
        for y in range(2, image.shape[1] + 1):
            # Generate submatrix and verify if all elements are equal to kernel
            sub_matrix = image_pad[x-1:x+2, y-1:y+2]
            dilated[x-2, y-2] = np.array(1 if (sub_matrix == structuring_kernel).any() else 0)

    return dilated


# Perform opening morphological transformation
def opening(image):
    eroded = erode_image(image)
    return dilate_image(eroded)


# Perform closing morphological transformation
def closing(image):
    dilated = dilate_image(image)
    return erode_image(dilated)


# Create masks 1 and 2 using given specification
def generate_masks(grayscale, limiarized):
    mask1 = grayscale * (1 - limiarized)
    mask2 = grayscale * limiarized
    return mask1.astype(int), mask2.astype(int)


# Find co-occurence matrix
def create_co_occurence_matrix(image, q):
    image_max = np.max(image)
    co_occurence_matrix = np.zeros([image_max + 1, image_max + 1], np.uint8)
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            # Calculate difference between seed and Q and update increment respective cell
            diff = np.abs(int(image[x, y]) - int(image[x + q[0], y + q[1]]))
            co_occurence_matrix[diff, image[x, y]] += 1
    # Return normalized matrix
    return (co_occurence_matrix / np.sum(co_occurence_matrix))


""" Methods to find Halaric Descriptors """
def auto_correlation(matrix, row, col):
    return np.sum(row * col * matrix)


def contrast(matrix, row, col):
    return np.sum(np.square(row - col) * matrix)


def dissimilarity(matrix, row, col):
    return np.sum(np.abs(row - col) * matrix)


def energy(matrix):
    return np.sum(np.square(matrix))


def entropy(matrix):
    matrix = matrix[matrix != 0]
    return np.sum(-matrix * np.log(matrix))


def homogeneity(matrix, row, col):
    return np.sum(matrix / (1 + np.square(row - col)))


def inverse_difference(matrix, row, col):
    return np.sum(matrix / (1 + np.abs(row - col)))


def max_prob(matrix):
    return np.max(matrix)


# Compute all descriptors for a given matrix
def compute_haralick_descriptors(matrix):
    # Store all descriptors
    descriptors = []
    
    row = np.zeros([matrix.shape[0], matrix.shape[1]])
    col = np.zeros([matrix.shape[0], matrix.shape[1]])
    for i in range(matrix.shape[0]):
        row[i, :] = i
        col[:, i] = i
    
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
        distances.append(np.sqrt(np.sum(np.square(query - image))))
    similarities = distances / np.max(distances)
    return similarities


if __name__ == "__main__":
    # Read user input
    index        = int(input().rstrip())
    q_value      = [int(q) for q in input().rstrip().split(" ")]
    f_param      = int(input().rstrip())
    t_param      = int(input().rstrip())
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
        limiarized_images.append(limiarization(gray_images[-1], t_param))  # Limiarize image

        # Perform morphological operation
        if f_param == 1:
            transformed_images.append(opening(limiarized_images[-1]))
        else:
            transformed_images.append(closing(limiarized_images[-1]))

        # Calculate masks
        masks = generate_masks(gray_images[-1], transformed_images[-1])
        masks1.append(masks[0])
        masks2.append(masks[1])

        descriptors.append(get_descriptors(masks1[-1], masks2[-1], q_value))  # Calculate all descriptors

    # Calculate similarities and order images
    similarities = get_ranked_similarities(descriptors[index], descriptors)
    order = [x for _, x in sorted(zip(similarities, image_names))]

    # Print output
    print(f"Query: {query_image_name}")
    print("Ranking:")
    for index, image_name in enumerate(order):
        print(f"({index}) {image_name}")
