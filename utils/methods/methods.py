import numpy as np
from scipy.signal import hilbert
import math


def calcInstaPhaseNorm(signal):

    insta_phase = hilbert(signal)
    insta_phase_norm = insta_phase/(2*math.pi)  # math.fmod(insta_phase/(2*math.pi), 1.)

    return insta_phase_norm


def calcPhaseResetIdx(v, t_k, phi_j):

    check11 = phi_j[t_k]
    # check12 = i*v*2*pi*phi_j[t_k]
    # check13 = exp(1i*v*2*pi*phi_j[t_k])
    check14 = np.mean(v*check11)  # mean(exp(1i*v*2*pi*phi_j[t_k]))
    phase_index = abs(check14)  # abs(mean(exp(1i*2*pi*phi_j[t_k])))

    return phase_index
