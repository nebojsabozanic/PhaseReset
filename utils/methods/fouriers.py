import numpy as np
from scipy.fftpack import fft


def power_spectrum_fft(signal, fs):

    L = len(signal)
    yf = fft(signal)
    T = 1 / fs
    xf = np.linspace(0.0, 1.0 / (2.0 * T), L // 2)
    P1 =  2.0 / L * np.abs(yf[0:L // 2])

    return P1, xf
