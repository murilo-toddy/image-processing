import numpy as np
import imageio


if __name__ == "__main__":
    # Read user input
    index = int(input().rstrip())
    q_value = [int(q) for q in input().rstrip().split(" ")]
    f_param = int(input().rstrip())
    t_param = int(input().rstrip())
    dataset_size = int(input().rstrip())
    image_names = []
    for i in range(dataset_size):
        image_names[i] = input().rstrip()
