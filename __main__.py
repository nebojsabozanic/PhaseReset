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
from utils.disp.showphases import showphases, show_signal, show_windows, showFFT
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

    # Write arguments to a text file
    write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    # store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    print('Output directory: %s' % output_dir)
    print('Log directory: %s' % log_dir)

    channels, fs, stims = readchannels()

    cz = channels[255, :]
    stims = stims[1, :]  # [0:6]
    stims = stims[2:]

    if (0):
        dur = 10000
        cz = cz[0:10000]
        stims = stims[stims < dur]

    show_signal(cz)
    # P1, xf = calcFFT(cz, fs)
    # showFFT(P1, xf)
    # tf = calc_stft(cz)
    # show_tf(tf)
    # show_windows(cz, stims, fs)

    # notch
    order = 6
    cutoff = 3.667
    y1 = butter_filter(cz, cutoff, fs, order)
    f0 = 60.
    Q = 30.
    b, a = signal.iirnotch(2 * f0 / fs, Q)
    y2 = signal.filtfilt(b, a, y1)
    P1, xf = calcFFT(y2, fs)
    showFFT(P1, xf)

    insta_phase_norm = calcInstaPhaseNorm(y2)
    show_signal(insta_phase_norm)
    # coeffs = calcPhaseResetIdx(1, stims, insta_phase_norm)
    coeffswin = calcPhaseResetIdxWin(1, stims, insta_phase_norm, 100, 100)

    stop = 1

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