import numpy as np
from scipy.signal import hilbert
import math


def calcInstaPhaseNorm(signal):

    y = hilbert(signal)
    angles = np.angle(y)
    insta_phase = np.unwrap(angles)
    insta_phase_norm = (insta_phase + math.pi)/(2*math.pi) % 1. # math.fmod(insta_phase/(2*math.pi), 1.)

    return insta_phase_norm


def calcPhaseResetIdx(v, t_k, phi_j):

    step1 = phi_j[t_k.astype(int)]
    #!!!! check with Peter!! it seems that in his paper phi goes from 0 to 2pi not -pi to pi as it is a convention
    step2 = 1j*v*2*math.pi*phi_j[t_k.astype(int)]
    step3 = np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)])
    step4 = np.mean(np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))  # mean(exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))
    phase_index = np.abs(np.mean(np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)])))  # abs(mean(exp(1j*2*math.pi*phi_j[t_k.astype(int)])))

    return phase_index
