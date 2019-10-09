"""Pipeline for the tinnitus analysis###"""
##############################################################
# 1. load .mat, .set, .mff eventually connect with neuropype

# 2. phase reset coefficient as described in ...

# (emd)

# plot 2d
############################################################
# Copyright (c) 2019 Nebojsa Bozanic, Peter Tass
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import argparse

from six import iteritems
from subprocess import Popen, PIPE
from utils.io.read import readchannels
from utils.methods.phasereset import calcPhaseResetIdx, calcPhaseResetIdxWin, calcInstaPhaseNorm
from utils.methods.fouriers import calcFFT
from utils.methods.n1p1 import n1p1, rerefAll, n1p1c
from utils.disp.showphases import showphases, show_signal, show_windows, showFFT, show_2signals, show_insta_phase, show_csignals
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
import time


def main(args):

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    if ~os.path.isabs(args.output_base_dir):
        dirpath = os.path.dirname(__file__)
        args.output_base_dir = os.path.join(dirpath, args.output_base_dir)
    output_dir =     os.path.join(os.path.expanduser(args.output_base_dir), subdir)
    if not os.path.isdir(output_dir):  # Create the model directory if it doesn't exist
        os.makedirs(output_dir)
    log_dir = args.logs_base_dir
    if ~os.path.isabs(args.logs_base_dir):
        log_dir = os.path.join(output_dir, args.logs_base_dir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)

    # Write arguments to a text file
    write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    # store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    print('Output directory: %s' % output_dir)
    print('Log directory: %s' % log_dir)

    channels, fs, stims = readchannels()

    cutit = 0

    classes = stims[0, :]
    classes = classes[2:]

    uc, uc_ind = np.unique(classes, return_inverse=True)
    len_uc = len(uc)
    # unique
    stims = stims[1, :]  # [0:6]
    stims = stims[2:]

    if (cutit):
        dur = 100000
        stims = stims[stims < dur]
        classes = classes[stims < dur]


    for cnt in range(256):
    # cnt = 222
    #if(1):
        cz = channels[cnt, :]
        if (len(cz) % 2):
            cz = cz[:-1]
        t = np.arange(0, len(cz)/fs, 1/fs)
        t1 = np.arange(0, 2000 / fs, 1 / fs)
        lmbd = 9
        test = np.exp(-lmbd * t1)
        test1 = 0.55*np.sin(2 * np.pi * 13 * t1)
        test2 = test1*test
        #show_signal(test2)
        phase0 = np.zeros(len(cz))
        aJump = 10*np.ones(len(cz)) #np.random.rand(1, len(stims)) add some noise
        #aJump = aJump[0]
        phase = phase0
        cz1 = np.zeros(len(cz))
        for cnt, i in enumerate(stims):
            temp = i[0].astype(int)
            jitter = random.randint(1, 100) - 25
            # phase[temp[0]+jitter:temp[0]+jitter+2000] = test
            cz1[temp[0]+jitter:temp[0]+jitter+2000] = test2
        # phase = np.cumsum(phaseR)
        #phase = np.convolve()
        #show_signal(cz1)
#        cz = np.sin(2 * np.pi * 13 * t + phase)
        # show_signal(cz)

        noise = 0.1*np.random.randn(1, len(t))

        cz1 += noise[0]

        #cz = cz1
        show_signal(cz)

        if (cutit):
            cz = cz[0:dur]
            stims = stims[stims < dur]

        # re - referencing
        #czr = reref(cz, channels)

        czr = cz
        if (0):
            meanref = rerefAll(channels)
            czr = cz - meanref
            #show_signal(czr)

        # covariance matrix

        # check it out DSS

        # std of boostrap

        # P1, xf = calcFFT(cz, fs)
        # showFFT(P1, xf)
        # P1m = 20 * np.log10(P1 / max(P1))
        # showFFT(P1m, xf)
        # tf = calc_stft(cz)
        # show_tf(tf)
        # show_windows(cz, stims, fs)

        # # notc
        order = 6
        cutoff = 3.667
        y1 = butter_filter(czr, cutoff, fs, order)

        # P1, xf = calcFFT(y1, fs)
        # P1m = 20 * np.log10(P1 / max(P1))
        # showFFT(P1m, xf)

        if (0):
            f0 = 60.
            Q = 10.
            b, a = signal.iirnotch(2 * f0 / fs, Q)
            y2 = signal.filtfilt(b, a, y1)
            # show_signal(y2)
            # P1, xf = calcFFT(y2, fs)
            # P1m = 20 * np.log10(P1 / max(P1))
            # showFFT(P1m, xf)
        else:
            # show_signal(y1)
            y2 = Implement_Notch_Filter(1000., 0.25, 60., 5., 3, 'butter', y1)

            # show_signal(y2)
            # P1, xf = calcFFT(y2, fs)
            # P1m = 20 * np.log10(P1 / max(P1))
            # showFFT(P1m, xf)

        # insta_phase_norm = calcInstaPhaseNorm(y2)
        # show_signal(insta_phase_norm)
        # coeffs = calcPhaseResetIdx(1, stims, insta_phase_norm)
        # coeffswin = calcPhaseResetIdxWin(1, stims, insta_phase_norm, 100, 100)

        # ave_y2_500, std_y2_500, ave_y2_1000, std_y2_1000 = n1p1(y2, stims, 400, 2000)
        ave_y2, std_y2 = n1p1c(y2, stims, 400, 2000, classes, uc, uc_ind, len_uc)
        # print(np.size(ave_y2))

        #show_2signals(ave_y2_500, ave_y2_1000, output_dir, cnt)
        show_csignals(ave_y2, output_dir, cnt)

    # y2 = y1 ## cz #!!!! omit later
        insta_phase_norm = calcInstaPhaseNorm(y2)

    #    show_insta_phase(insta_phase_norm)
        coeffswin = calcPhaseResetIdxWin(1, stims, insta_phase_norm, 400, 1000)
        show_signal(coeffswin)
        # show_2signals(std_y2_500, std_y2_1000)

        #temp = stats.zscore(ave_y2_500)
        # print(temp)
        #print(max(temp))
        #print(np.argmax(temp))
        #print(len(temp))
        #show_signal(ave_y2_500)
        #show_signal(temp)

    stop = 1


# Required input defintions are as follows;
# time:   Time between samples
# band:   The bandwidth around the centerline freqency that you wish to filter
# freq:   The centerline frequency to be filtered
# ripple: The maximum passband ripple that is allowed in db
# order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
#         IIR filters are best suited for high values of order.  This algorithm
#         is hard coded to FIR filters
# filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
# data:         the data to be filtered
def Implement_Notch_Filter(fs, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter, lfilter
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data



def store_revision_info(src_path, output_dir, arg_string):
    # Get git hash
    cmd = ['git', 'rev-parse', 'HEAD']
    try:
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror

    cmd = ['git', 'diff', 'HEAD']
    try:
        # Get local changes
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        # text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def butter(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_filter(data, cutoff, fs, order=5):
    b, a = butter(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='log')
    parser.add_argument('--output_base_dir', type=str,
                        help='Directory where to write output images.', default='output')

    parser.add_argument('--methods', type=list,
                        help='which methods are applied', default=['phase reset'])

    parser.add_argument('--data_path', type=str,
                        help='Folder and files that will be analyzed', default=os.path.join('data', 'data1.mat'))

    parser.add_argument('--freq_lim', type=float,
                        help='for a more pleasent display', default=300)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))