from scipy.io import loadmat
import numpy as np
import random
import os
import time

def readchannels(args):

    # select with mouse?
    experiment_list = ['20190919_010308',
                       '20190919_010845',
                       '20191008_062255',
                       '20191011_043914',
                       '20191011_044342',
                       '20191011_044634',
                       '20191021_023046',
                       '20191021_023358',
                       '20191021_024940',
                       '20191208_054800',
                       '20191208_060149',
                       '20191208_062007',
                       '20191208_063407',
                       '20191208_065528',
                       '20191208_070804']

    # at some point channel_names will be read from the name of the experiment since they are redundant
    channel_list = ['a1_20190919_010308mff2',
                    'a2_20190919_010845mff2',
                    'Tina10819round21_20191008_062255mff2',
                    'a1_20191011_043914nebosoftclicksmff2',
                    'a1_20191011_044342neboloudclicksmff2',
                    'a1_20191011_044634nebo1sclicklessmff2',
                    'a1_20191021_023046_clickmff2',
                    'a2_20191021_023358_noclickmff2',
                    'a3_20191021_024940_singlefilemff2',
                    'a2_20191208_054800_NeboERB1mff2',
                    'a2_20191208_060149_NeboERB2mff2',
                    'a2_20191208_062007_NeboERB4mff2',
                    'a2_20191208_063407_20msrampmff2',
                    'a2_20191208_065528_70msrampmff2',
                    'a2_20191208_070804_20msshufflemff2']

    # Python no (1 is 2, 0 is 1)
    exp_no = 12  # argument!!
    args.experiment = experiment_list[exp_no]

    args.filename = 'data/'     + experiment_list[exp_no]
    args.channel_name = channel_list[exp_no]

    start = time.time()
    data = loadmat(args.filename)
    print(time.time() - start)
    args.channels = data[args.channel_name]

    args.channels = args.channels[args.select_ch - 1, :]
    args.fs = data['EEGSamplingRate'][0][0]
    args.stims = data['evt_ECI_TCPIP_55513']

    args.times = args.stims[1, 2:]
    args.times = args.times.astype(int)
    # temp until figured out how to have it more elegantly
    args.flag = 0
    if os.path.exists(args.filename + '_times.mat'):
        times_all = loadmat(args.filename + '_times.mat')
        offset = args.stims[1, 1]
        offset = offset[0]
        # boundary
        boundary = np.array([1.916, 1.916, 1.95, 1.995, 1.9, 1.986])
        # onset
        onset = np.array([2.5425, 2.5355, 2.6103, 2.6127, 2.5196, 2.6156])
        offset = onset[exp_no - 9] * args.fs # add boundary
        offset = offset.astype(int)
        relative = times_all['times']
        relative = relative[0]
        relative *= args.fs
        relative = relative.astype(int)
        args.times = offset + relative
        args.stimuli = np.arange(1000, 12000, 2000)  # np.array(1000)
        args.reps = 50  # 300
        repmat = np.tile(args.stimuli, (args.reps, 1))
        repmat_transposed = repmat.transpose()
        args.classes = repmat_transposed.reshape(1, -1)
        args.flag = 1

        #args.times = args.times[0:50]
        ##fixargs.classes = args.classes[0:50]
        #args.channels = args.channels[:, args.times[0]-101:args.times[10]+2001]
        #args.times = args.times - args.times[0] + 101
    return args


def surro(args):

    args.experiment = 'Surrogates'
    # generate a fingerprint
    t_erp = np.arange(0, args.erp_len / args.sfs, 1 / args.sfs)
    exponential_decay = np.exp(-args.lambd * t_erp)
    erp_osc = args.amp * np.sin(2 * np.pi * args.erp_freq * t_erp)
    args.erp = erp_osc * exponential_decay

    args.signal_len = args.channels.shape[-1]
    # put fingerprints at adequate times
    t_signal = np.arange(0, args.signal_len / args.sfs, 1 / args.sfs)
    args.signal_clean = np.zeros([1, len(t_signal)])
    for cnt, ts in enumerate(args.times):
        print(ts)
        jitter = random.randint(1, 2*args.jit) - args.jit  # introduce a parameter
        args.signal_clean[0, ts+jitter:ts+jitter+args.erp_len] = args.erp[0:args.erp_len]

    args.noise = args.noise_amp * np.random.randn(1, len(t_signal))

    args.channels = np.delete(args.channels, np.s_[1:], 0)
    args.channels = []
    args.channels = args.signal_clean + args.noise[0]

    return args
