import numpy as np
import imageio


def luminance_weights(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def limiarization(image):
    return image


def erode_image(image):
    return image


def dilate_image(image):
    return image


def opening(image):
    eroded_image = erode_image(image)
    return dilate_image(eroded_image)


def closing(image):
    dilated_image = dilate_image(image)
    return erode_image(dilated_image)


def generate_masks(image):
    return image, image


if __name__ == "__main__":
    # Read user input
    index = int(input().rstrip())
    q_value = [int(q) for q in input().rstrip().split(" ")]
    f_param = int(input().rstrip())
    t_param = int(input().rstrip())
    dataset_size = int(input().rstrip())
    images = []
    for i in range(dataset_size):
        if i == index:
            image = imageio.imread(input().rstrip())
            continue
        images.append(imageio.imread(input().rstrip()))

    gray_image = luminance_weights(image)
    limiarized_image = limiarization(gray_image)
    if f_param == 1:
        transformed_image = closing(limiarized_image)
    else:
        transformed_image = opening(limiarized_image)

    mask1, mask2 = generate_masks(transformed_image)
