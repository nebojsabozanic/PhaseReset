import numpy as np
from scipy.fftpack import fft


def power_spectrum_fft(signal, fs):

    length = len(signal)
    yf = fft(signal)
    t = 1 / fs
    xf = np.linspace(0.0, 1.0 / (2.0 * t), length // 2)
    p1 = 2.0 / length * np.abs(yf[0:length // 2])

    return p1, xf
