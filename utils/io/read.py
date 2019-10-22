from scipy.io import loadmat
import numpy as np
import random


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
                       '20191021_024940']

    # at some point channel_names will be read from the name of the experiment since they are redundant
    channel_list = ['a1_20190919_010308mff2',
                    'a2_20190919_010845mff2',
                    'Tina10819round21_20191008_062255mff2',
                    'a1_20191011_043914nebosoftclicksmff2',
                    'a1_20191011_044342neboloudclicksmff2',
                    'a1_20191011_044634nebo1sclicklessmff2',
                    'a1_20191021_023046_clickmff2',
                    'a2_20191021_023358_noclickmff2',
                    'a3_20191021_024940_singlefilemff2']

    # Python no (1 is 2, 0 is 1)
    exp_no = 6  # argument!!
    args.experiment = experiment_list[exp_no]

    args.filename = 'data/' + experiment_list[exp_no]
    args.channel_name = channel_list[exp_no]

    data = loadmat(args.filename)
    args.channels = data[args.channel_name]

    args.fs = data['EEGSamplingRate'][0][0]
    args.stims = data['evt_ECI_TCPIP_55513']

    times_all = loadmat('data/20191021_024940_times.mat')
    args.times_truth = times_all['times']

    return args


def surro(args):

    args.experiment = 'Surrogates'
    # generate a fingerprint
    t_erp = np.arange(0, args.erp_len / args.sfs, 1 / args.sfs)
    exponential_decay = np.exp(-args.lambd * t_erp)
    erp_osc = 1. * np.sin(2 * np.pi * args.erp_freq * t_erp)
    args.erp = erp_osc * exponential_decay

    # put fingerprints at adequate times
    t_signal = np.arange(0, args.signal_len / args.sfs, 1 / args.sfs)
    args.signal_clean = np.zeros([1, len(t_signal)])
    for cnt, i in enumerate(args.times):
        ts = i[0].astype(int)
        ts = ts[0]
        jitter = random.randint(1, 2*args.jit) - args.jit  # introduce a parameter
        args.signal_clean[0, ts+jitter:ts+jitter+args.erp_len] = args.erp[0:args.erp_len]

    args.noise = args.noise_amp * np.random.randn(1, len(t_signal))

    args.channels = np.delete(args.channels, np.s_[1:], 0)
    args.channels = []
    args.channels = args.signal_clean + args.noise[0]

    return args
