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
from utils.io.read import readchannels, surro
from utils.methods.phasereset import getPhaseResetIndices #calcPhaseResetIdx, calcPhaseResetIdxWin, calcInstaPhaseNorm, calcPhaseResetIdxWin_c
# from utils.methods.fouriers import calcFFT
from utils.methods.n1p1 import getN1P1  # n1p1, rerefAll, n1p1c
# from utils.disp.showphases import showphases, show_signal, show_windows, showFFT, show_2signals, show_insta_phase, show_csignals
from utils.methods.process import proc, getStats
# from scipy import signal
import time
from utils.disp.showphases import show_examples

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

    # read real or generate surrogate data
    # solve this issue
    args = readchannels(args)  #
    if args.surro:
        args.times = args.stims[1, 2:]  #
        args = surro(args)


    # Write arguments to a text file
    write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # clean the data, and filter (highpass, lowwpass, notch, eyeblinks, eyemovements, headmovements...)
    #start = time.time()
    #args = proc(args)
    #print(time.time() - start)

    args.times = args.stims[1, 2:]
    args = getStats(args) # put in output

    args.times = args.times_truth
    # add a progress bar
    args = getN1P1(args)

    args = getStats(args) # put in output

    # show_examples(args)

    # get the instantaneous phases and their phase reset indices
    args = getPhaseResetIndices(args)


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

    parser.add_argument('--win_l', type=int,
                        help='left part of the window', default=100)
    parser.add_argument('--win_r', type=int,
                        help='left part of the window', default=900)
    parser.add_argument('--if_data_large', type=bool,
                        help='if the memory is not large enough', default=0)
    parser.add_argument('--dur', type=int,
                        help='duation in case the memory is not too large', default=10000)

    parser.add_argument('--do_reref', type=bool,
                        help='average rereferencing', default=1)

    parser.add_argument('--surro', type=bool,
                        help='generate artificial data', default=0)

    parser.add_argument('--sfs', type=float,
                        help='surro sampling frequency', default=1e3)
    parser.add_argument('--erp_len', type=int,
                        help='duration of erp fingerprint in samples', default=1347)
    parser.add_argument('--lambd', type=float,
                        help='exponential decay parameter (strength)', default=9.)
    parser.add_argument('--erp_freq', type=float,
                        help='Frequency of the erp', default=7.)
    parser.add_argument('--signal_len', type=int,
                        help='generate artificial data', default=5e5)
    parser.add_argument('--jit', type=int,
                        help='generate artificial data', default=2)
    parser.add_argument('--noise_amp', type=float,
                        help='generate artificial data', default=0.2)


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))