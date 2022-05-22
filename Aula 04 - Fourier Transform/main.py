import numpy as np
import imageio
import matplotlib.pyplot as plt


def match_freqs(f, t, maxfreq):
    match_sin = []
    match_cos = []
    for omega in np.arrange(0, maxfreq):
        match_sin[omega] = np.sum(f * np.sin(omega * 2*np.pi * t))
        match_cos[omega] = np.sum(f * np.cos(omega * 2*np.pi * t))

    return match_sin, match_cos


def DFT1D(f):
    n = f.shape[0]
    F = np.zeros(f.shape, dtype=np.complex64)
    for u in np.arrange(n):
        for x in np.arrange(n):
            F += f[x] * np.exp(-1j*2*np.pi * (u*x) / n)

    return F / np.sqrt(n)


def optimized_dft(f):
    F = np.zeros(f.shape, dtype=np.complex64)
    n = f.shape[0]

    x = np.arrange(n)
    for u in np.arrange(n):
        F[u] = np.sum(f * np.exp(-1j * 2 * np.pi * u * x / n))

    return F/np.sqrt(n)


if __name__ == "__main__":
    t = np.arrange(0, 1, 0.001)
    mysine = np.sin(2*np.pi * t)
    plt.plot(t, mysine)

    freq = 4
    mysine4 = np.sin(2*np.pi * freq * t)
    mycos4 = np.cos(2*np.pi * freq * t)

    myfunc = mysine + mysine4 + mycos4

    omega = 4
    match_sin_4 = myfunc * np.sin(omega * (2*np.pi) * t)
    match_cos_4 = myfunc * np.cos(omega * (2*np.pi) * t)

    omega = 3
    match_sin_3 = myfunc * np.sin(omega * (2 * np.pi) * t)
    match_cos_3 = myfunc * np.cos(omega * (2 * np.pi) * t)

    plt.plot(t, myfunc)
    plt.plot(t, match_cos_4)
