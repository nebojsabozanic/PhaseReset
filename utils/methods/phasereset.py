import numpy as np
from scipy.signal import hilbert
import math
from utils.disp.showphases import showphases, show_signal, show_csignals
import time
from scipy.fftpack import fft
import matplotlib.pyplot as plt

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
        check0 = i[0].astype(int) - win_l
        check1 = i[0].astype(int) + win_r
        step1 = phi_j[check0[0] : check1[0]]
        #show_signal(step1)
        #!!!! check with Peter!! it seems that in his paper phi goes from 0 to 2pi not -pi to pi as it is a convention
        step2 = 1j*v*2*math.pi*step1
        step3 = np.exp(step2)
        # show_signal(step3)
        step3_all += step3
        # step4 = np.mean(np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))  # mean(exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))

    # phase_index = np.mean(step3_all)  # abs(mean(exp(1j*2*math.pi*phi_j[t_k.astype(int)])))
    temp = len(t_k)
    step3_all /= len(t_k)
    phase_index = np.abs(step3_all)
    # showphases(phase_index)

    #print(np.abs(phase_index))

    return phase_index


def calcPhaseResetIdxWin_c(v, phi_j, times, wind_l, wind_r, uc, uc_ind, len_uc):

    wind_ = np.zeros([len(times), wind_l + wind_r])
    for cnti, i in enumerate(times):

        i1 = i[0].astype(int)
        wind_[cnti, :] = phi_j[i1[0] - wind_l : i1[0] + wind_r] # faster

    #print(len_uc)
    ave_step3 = np.zeros([len_uc, wind_l + wind_r])
    std_step3 = np.zeros([len_uc, wind_l + wind_r])

    step2 = 1j*v*2*math.pi*wind_
    step3 = np.exp(step2)

    for i, uclass in enumerate(uc):
        ind = (uc_ind == i)
        # print(i)
        # print(ind)
        ave_step3[i, :] = np.mean(step3[ind, :], 0)
        std_step3[i, :] = np.std(step3[ind, :], 0)

    step3_all = 0
    for i in times:
        check0 = i[0].astype(int) - wind_l
        check1 = i[0].astype(int) + wind_r
        step1 = phi_j[check0[0] : check1[0]]
        #show_signal(step1)
        #!!!! check with Peter!! it seems that in his paper phi goes from 0 to 2pi not -pi to pi as it is a convention
        step2 = 1j*v*2*math.pi*step1
        step3 = np.exp(step2)
        # show_signal(step3)
        step3_all += step3
        # step4 = np.mean(np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))  # mean(exp(1j*v*2*math.pi*phi_j[t_k.astype(int)]))

    # phase_index = np.mean(step3_all)  # abs(mean(exp(1j*2*math.pi*phi_j[t_k.astype(int)])))

    phase_index = np.abs(ave_step3)
    # showphases(phase_index)

    #print(np.abs(phase_index))

    return phase_index

def getPhaseResetIndices(args):

    cnt = 133
    y2 = args.channels[cnt,:] ## singled_out_filtered_notched[cnt, :]
    show_signal(y2)
    # insta_phase_norm = calcInstaPhaseNorm(y2)
    # show_signal(insta_phase_norm)
    # coeffs = calcPhaseResetIdx(1, stims, insta_phase_norm)
    # coeffswin = calcPhaseResetIdxWin(1, stims, insta_phase_norm, 100, 100).
    # y2 = y1 ## cz #!!!! omit later
    insta_phase_norm = calcInstaPhaseNorm(y2)
    # show_phases(insta_phase_norm)

    #    show_insta_phase(insta_phase_norm)
    coeffswin = calcPhaseResetIdxWin(1, args.times, insta_phase_norm, args.win_l, args.win_r)
    show_signal(coeffswin)
    coeffswin = calcPhaseResetIdxWin_c(1, insta_phase_norm, args.times, args.win_l, args.win_r, args.uc, args.uc_ind, args.len_uc)
    show_csignals(coeffswin, args.output_dir, cnt)

    return args


def histogram_phases(insta_phases, times, wind_l, wind_r, uc, uc_ind, len_uc):
    wind_ = np.zeros([len(times), wind_l + wind_r])
    for cnti, i in enumerate(times):
        i1 = i[0].astype(int)
        wind_[cnti, :] = insta_phases[i1[0] - wind_l: i1[0] + wind_r]  # faster

    # print(len_uc)
    nbin = 100
    hist_wind = np.zeros([len_uc, nbin, wind_l + wind_r])


    for i, uclass in enumerate(uc):
        ind = (uc_ind == i)
        temp = wind_[ind, :]
        print(temp.shape[0])
        for cnti in range(temp.shape[1]):
            test = np.histogram(temp[:, cnti], nbin, (0, 1)) # calc hist wind_[ind, :]
            hist_wind[i, :, cnti] = test[0]
        stop = 1

    testimage = np.squeeze(hist_wind[1, :, :])
    #testimage = testimage[-1:0:-1,:]
    plt.imshow(testimage)
    plt.show()
    return hist_wind