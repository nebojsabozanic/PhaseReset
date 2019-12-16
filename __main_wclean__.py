"""Pipeline for the tinnitus analysis###"""

# Copyright (c) 2019 Nebojsa Bozanic, Tina Munjal, Peter Tass
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

import math
from six import iteritems
from subprocess import Popen, PIPE
from utils.methods.phasereset import histogram_phases
from scipy.io import loadmat #
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation
# from scipy.fftpack import fft
from scipy.signal import hilbert
from scipy import signal

def main(args):

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    if ~os.path.isabs(args.output_base_dir):
        dirpath = os.path.dirname(__file__)
        args.output_base_dir = os.path.join(dirpath, args.output_base_dir)
    output_dir = os.path.join(os.path.expanduser(args.output_base_dir), subdir)
    if not os.path.isdir(output_dir):  # Create the model directory if it doesn't exist
        os.makedirs(output_dir)
    log_dir = args.logs_base_dir
    if ~os.path.isabs(args.logs_base_dir):
        log_dir = os.path.join(output_dir, args.logs_base_dir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    # store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    print('Output directory: %s' % output_dir)
    print('Log directory: %s' % log_dir)

    args.output_dir = output_dir

    #data = loadmat('data/20msclean_n.mat')
    data = loadmat('data/1ERBclean_n.mat')

    args.channels = data['channels']
    args.labels = data['labels']
    args.labels = args.labels[0]

    cz = np.squeeze(args.channels[-1, :, :])
    ind = (args.labels == 1000)
    cz_ind = cz[:, ind]
    x = np.arange(-100,2000)
    #plt.plot(x, np.mean(cz_ind, 1)) #, 'LineWidth', 2
    #plt.show()
    ##plot([0, 0], [-4, 4], 'LineWidth', 2)
    #plot([-100, 1999], [0, 0], 'LineWidth', 2)
    #xlim([-100 1000])
    #legend(num2str([1: 69]'))

    fs = 1000
    emd = EMD()
    n_tr = cz_ind.shape[1]
    nbin = 200
    hist_wind = np.zeros([nbin, cz_ind.shape[0]])
    wind_ = np.zeros([cz_ind.shape[0], cz_ind.shape[1]])
    for cnt_trial in range(0, n_tr):

        S = np.squeeze(cz_ind[:, cnt_trial])

        order = 2
        lf_cutoff = 1.
        hf_cutoff = 4.
        # imf3 = butter_bandpass(S, lf_cutoff, hf_cutoff, fs, order)

        emd.emd(S)
        imfs, res = emd.get_imfs_and_residue()
        imf3 = imfs[-3]

        #vis = Visualisation()
        #t = np.arange(0, S.size/fs, 1/fs)
        #vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
        #vis.plot_instant_freq(t, imfs=imfs)
        #vis.show()
        stop = 1

        # check the emd components
        if 1:
        #for imf in imfs:
            spectrum = plt.magnitude_spectrum(imf3, fs)
            #plt.plot(spectrum[1], spectrum[0])
            #plt.show()
            #stop = 1

        #plt.plot(imf3)
        #plt.show()

        # let's say the algorithm above calculates the spectrum magnitude and gives us the imf which is in delta range
        # manually we have seen it is imfs[-3]

        y2 = imf3
        y = hilbert(y2)
        angles = np.angle(y)
        insta_phase = np.unwrap(angles)  # should we ingore this and go straight to the normsss
        insta_phase_norm = (insta_phase + math.pi) / (2 * math.pi) % 1.

        wind_[:, cnt_trial] = insta_phase_norm
        print(cnt_trial)
        stop = 1

    #phase_reset_wind = np.zeros([args.len_uc, args.nbin, args.win_l + args.win_r])

    v = 1
    #phase_reset_wind = np.exp(1j * v * 2 * math.pi * wind_)

    #phase_reset_mean = np.zeros([1, cz_ind.shape[0]])
    #phase_reset_std = np.zeros([1, cz_ind.shape[0]])

    #for i, uclass in enumerate(args.uc):
    #ind = (args.uc_ind == i)
    #temp = wind_[ind, :]
    for cnti in range(wind_.shape[0]):
        test = np.histogram(wind_[cnti, :], nbin, (0, 1))  # calc hist wind_[ind, :]
        hist_wind[:, cnti] = test[0]

    # step = np.abs(np.mean(phase_reset_wind[ind, :]))
    # mean_step = np.mean()

    #phase_reset_mean[i, :] = np.abs(np.mean(phase_reset_wind[ind, :], 0))
    #phase_reset_std[i, :] = np.abs(np.std(phase_reset_wind[ind, :], 0))

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.imshow(hist_wind)  # , aspect='auto'
    plt.title(np.max(hist_wind))
    # ax.set_adjustable('box-forced')
    #filename = 'histophases' + str(cnt_ch) + 'ch' + str(i) + 'cl' + '.png'
    #plt.savefig(os.path.join(args.output_dir, filename), bbox_inches='tight', pad_inches=0)
    #plt.close()
    plt.show()
    # plt.waitforbuttonpress(0.1)
    # save_ call show (methods>disp)

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


def butter_bandpass(data, lowcut, highcut, fs, order=5):
    b, a = butter_filter_band(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_filter_band(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str,
                        help='Experiment name', default='')

    parser.add_argument('--channel_name', type=str,
                        help='Experiment name', default='')

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='log')
    parser.add_argument('--output_base_dir', type=str,
                        help='Directory where to write output images.', default='output')

    parser.add_argument('--surro', type=bool,
                        help='generate artificial data', default=0)

    parser.add_argument('--show_midplots', type=bool,
                        help='plot data between the processing steps', default=0)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
