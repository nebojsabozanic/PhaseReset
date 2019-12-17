import numpy as np
from scipy.signal import hilbert
import math
from utils.disp.showphases import show_signal, show_csignals  # showphases
from scipy.fftpack import fft #, hilbert
import matplotlib.pyplot as plt
import os
from numba import jit
import scipy as sp
import scipy.ndimage

#@jit(nopython=True)
def calc_insta_phase_norm(signal):
    print('in')
    y = hilbert(signal)
    print('out')
    angles = np.angle(y)
    insta_phase = np.unwrap(angles) # should we ingore this and go straight to the normsss
    insta_phase_norm = (insta_phase + math.pi)/(2*math.pi) % 1.

    return insta_phase_norm


def calc_phase_reset_idx(v, t_k, phi_j):

    phase_index = np.abs(np.mean(np.exp(1j*v*2*math.pi*phi_j[t_k.astype(int)])))

    return phase_index

def calc_phase_reset_win(v, signal):

    phase_index = np.exp(1j*v*2*math.pi*signal)

    return phase_index


# def calcPhaseResetIdxWin(v, t_k, phi_j, win_l, win_r):
#
#     step3_all = 0
#     for i in t_k:
#         check0 = i[0].astype(int) - win_l
#         check1 = i[0].astype(int) + win_r
#         step1 = phi_j[check0[0] : check1[0]]
#         step2 = 1j*v*2*math.pi*step1
#         step3 = np.exp(step2)
#         step3_all += step3
#
#     step3_all /= len(t_k)
#     phase_index = np.abs(step3_all)
#
#     return phase_index


def calcPhaseResetIdxWin_c(v, phi_j, times, wind_l, wind_r, uc, uc_ind, len_uc):

    wind_ = np.zeros([len(times), wind_l + wind_r])
    for cnti, i in enumerate(times):

        i1 = i[0].astype(int)
        wind_[cnti, :] = phi_j[i1[0] - wind_l : i1[0] + wind_r] # faster

    ave_step3 = np.zeros([len_uc, wind_l + wind_r])
    std_step3 = np.zeros([len_uc, wind_l + wind_r])

    step2 = 1j*v*2*math.pi*wind_
    step3 = np.exp(step2)

    for i, uclass in enumerate(uc):
        ind = (uc_ind == i)
        ave_step3[i, :] = np.mean(step3[ind, :], 0)
        std_step3[i, :] = np.std(step3[ind, :], 0)

    step3_all = 0
    for i in times:
        check0 = i[0].astype(int) - wind_l
        check1 = i[0].astype(int) + wind_r
        step1 = phi_j[check0[0] : check1[0]]
        step2 = 1j*v*2*math.pi*step1
        step3 = np.exp(step2)
        step3_all += step3

    phase_index = np.abs(ave_step3)

    return phase_index


def getPhaseResetIndices(args):

    cnt = 133
    y2 = args.channels[cnt,:]
    show_signal(y2)
    insta_phase_norm = calc_insta_phase_norm(y2)
    coeffswin = calcPhaseResetIdxWin(1, args.times, insta_phase_norm, args.win_l, args.win_r)
    show_signal(coeffswin)
    coeffswin = calcPhaseResetIdxWin_c(1, insta_phase_norm, args.times, args.win_l, args.win_r, args.uc, args.uc_ind, args.len_uc)
    show_csignals(coeffswin, args.output_dir, cnt)

    return args


def histogram_phases(args):

    #for cnt_ch in range(args.singled_out_filtered_notched.shape[0]):
    if 1:
        cnt_ch = 68
        y2 = args.singled_out_filtered_notched[cnt_ch, :]
        insta_phase_norm = calc_insta_phase_norm(y2)

        # show_insta_phase(insta_phase_norm)

        wind_ = np.zeros([len(args.times), args.win_l + args.win_r])
        for cnti, i in enumerate(args.times):
            i1 = int(i)
            wind_[cnti, :] = insta_phase_norm[i1 - args.win_l: i1 + args.win_r]  # faster

        # phase_reset_here

        # print(len_uc)
        args.nbin = 200
        hist_wind = np.zeros([args.len_uc, args.nbin, args.win_l + args.win_r])
        phase_reset_wind = np.zeros([args.len_uc, args.nbin, args.win_l + args.win_r])

        phase_reset_wind = calc_phase_reset_win(1, wind_)

        phase_reset_mean = np.zeros([args.len_uc, args.win_l + args.win_r])
        phase_reset_std = np.zeros([args.len_uc, args.win_l + args.win_r])

        for i, uclass in enumerate(args.uc):
            ind = (args.uc_ind == i)
            temp = wind_[ind, :]
            for cnti in range(temp.shape[1]):
                test = np.histogram(temp[:, cnti], args.nbin, (0, 1))  # calc hist wind_[ind, :]
                hist_wind[i, :, cnti] = test[0]

            # step = np.abs(np.mean(phase_reset_wind[ind, :]))
            # mean_step = np.mean()

            phase_reset_mean[i, :] = np.abs(np.mean(phase_reset_wind[ind, :], 0))
            phase_reset_std[i, :] = np.abs(np.std(phase_reset_wind[ind, :], 0))

            testimage = np.squeeze(hist_wind[i, :, :])
            sigma_y = 2.0
            sigma_x = 2.0
            sigma = [sigma_y, sigma_x]
            y = sp.ndimage.filters.gaussian_filter(testimage, sigma, mode='constant')
            # fig, ax = plt.subplots(nrows=1, ncols=1)
            plt.imshow(y)  # , aspect='auto'
            plt.title(np.max(y))
            #ax.set_adjustable('box-forced')
            filename = 'histophases' + str(cnt_ch) + 'ch' + str(i) + 'cl' + '.png'
            plt.savefig(os.path.join(args.output_dir, filename), bbox_inches = 'tight', pad_inches = 0)
            plt.close()
            # plt.show()
            # plt.waitforbuttonpress(0.1)
            # save_ call show (methods>disp)

        show_csignals(phase_reset_mean, phase_reset_std, args.output_dir, cnt_ch, 'phase_reset_indx')

    return args
