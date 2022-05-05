import numpy as np
import imageio
import matplotlib.pyplot as plt
import random


def histogram_equalization(A, no_levels):
    hist = histogram(A, no_levels)
    hist_cumul = np.zeros(no_levels)
    hist_cumul[0] = hist[0]
    for i in range(1, no_levels):
        hist_cumul[i] = hist[i] + hist_cumul[i-1]

    hist_transf = np.zeros(no_levels).astype(np.uint8)
    N, M = A.shape
    A_eq = np.zeros([N, M]).astype(np.uint8)

    for z in range(no_levels):
        s = hist_cumul[z] * (no_levels - 1) / float(M * N)
        hist_transf[z] = s

        A_eq[np.where(A == z)] = s

    return A_eq, hist_transf


def histogram(A, no_levels):
    hist = np.zeros(no_levels).astype(int)

    for i in range(no_levels):
        no_pixels_value_i = np.sum(A == i)
        hist[i] = no_pixels_value_i

    return hist


if __name__ == "__main__":
    z = np.arange(256)
    s_ident = z  # Identity
    s_inver = 255 - z  # Invert

    plt.figure(figsize=(6, 6))
    plt.plot(z, s_inver)
    plt.show()

    c = 255 / (np.log2(1 + 255))
    z_log2 = c * np.log2(z + 1)
    plt.plot(z, z_log2)
    