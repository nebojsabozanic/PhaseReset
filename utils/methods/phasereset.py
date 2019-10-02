import numpy as np
from scipy.signal import hilbert
import math
from utils.disp.showphases import showphases


def calcInstaPhaseNorm(signal):

    y = hilbert(signal)
    angles = np.angle(y)
    insta_phase = np.unwrap(angles)
    insta_phase_norm = (insta_phase + math.pi)/(2*math.pi) % 1.  # math.fmod(insta_phase/(2*math.pi), 1.)

    return insta_phase_norm


def calcPhaseResetIdx(v, t_k, phi_j):

    step1 = phi_j[t_k.astype(int)]
    #!!!! check with Peter!! it seems that in his paper phi goes from 0 to 2pi not -pi to pi as it is a convention
    step2 = 1j*v*2*math.pi*phi_j[t_k.astype(int)]
    step3 = np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)])
    showphases(step3)
    step4 = np.mean(np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))  # mean(exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))
    phase_index = np.abs(np.mean(np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)])))  # abs(mean(exp(1j*2*math.pi*phi_j[t_k.astype(int)])))
    print(phase_index)

    return phase_index


def calcPhaseResetIdxWin(v, t_k, phi_j, win_l, win_r):

    step3_all = 0
    for i in t_k:
        check0 = i[0].astype(int)-win_l
        check1 = i[0].astype(int) + win_r
        step1 = phi_j[check0[0]-win_l : check1[0]+win_r]
        #!!!! check with Peter!! it seems that in his paper phi goes from 0 to 2pi not -pi to pi as it is a convention
        step2 = 1j*v*2*math.pi*step1
        step3 = np.exp(step2)
        # showphases(step3)
        step3_all += step3
        # step4 = np.mean(np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))  # mean(exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))

    phase_index = np.mean(step3_all)  # abs(mean(exp(1j*2*math.pi*phi_j[t_k.astype(int)])))

    showphases(phase_index)

    print(np.abs(phase_index))

    return phase_index
