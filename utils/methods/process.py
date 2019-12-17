import numpy as np
from scipy import signal
# from utils.methods.fouriers import power_spectrum_fft
from utils.disp.showphases import show_fft, show_signal, magnospec
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PyEMD import EMD, Visualisation



def proc(args):

    args.singled_out = args.channels
    # show_signal(args.singled_out[0, :])
    # delete 0?
    # if (0):
    #     args.singled_out = args.singled_out[:, :-36]
    # for cnt in range(args.channels.shape[0]):
    #    test = args.channels[cnt, np.abs(args.channels[cnt, :]) < 1]

    if args.if_data_large:
        args.singled_out = args.singled_out[:, :args.duration]

    if len(args.channels[0, :]) % 2:
        args.singled_out = args.singled_out[:, :-1]  # potentially to all    args.singled_out_filtered = args.channels;

    if not args.flag:
        args.classes = args.stims[0, 2:]

    if args.if_data_large:
        dur1 = args.dur - args.win_r
        ind = args.stims[1, 2:] < dur1
        args.times = args.times[ind]
        args.classes = args.classes[ind]

    # problem with no classes or 1 class in case of one single file FIX BUGGG
    #if args.classes.size < 4
    #    args.classes = np.arrayprint

    args.uc, args.uc_ind = np.unique(args.classes, return_inverse=True)
    args.len_uc = len(args.uc)

    if not args.surro:

        # move this before the loop, take care, once you calc, then in the loop you take it out
        # re - referencing change to spatial
        # czr = reref(cz, channels)
        if args.do_reref:
            meanref = reref_all(args.singled_out)
            args.singled_out -= meanref
            # show_signal(singled_out)

        args.singled_out_filtered = args.singled_out
        args.singled_out_filtered_notched = args.singled_out_filtered

        for cnt in range(args.channels.shape[0]):

            if args.show_midplots:
                print('channel raw')
                args.temp_add = 'raw'
                show_signal(args.singled_out[cnt, :], args)
                magnospec(args.singled_out[cnt, :], args.fs)

            args.order = 2
            args.lf_cutoff = 1.
            args.hf_cutoff = 4.
            # args.singled_out_filtered[cnt, :] = butter_filter(args.singled_out[cnt, :], cutoff, args.fs, order)
            # add to the object, and always take the last (as in the cell)
            args.singled_out_filtered[cnt, :] = butter_bandpass(args.singled_out[cnt, :], args.lf_cutoff,
                                                                args.hf_cutoff, args.fs, args.order)

            if args.show_midplots:
                print('filtered')
                args.temp_add = 'filtered'
                show_signal(args.singled_out_filtered[cnt, :], args)
                magnospec(args.singled_out_filtered[cnt, :], args.fs)

            # no hard coded!!!
            args.notch_width = 0.5
            args.notch_freq = 60.
            args.singled_out_filtered_notched[cnt, :] = implement_notch_filter(args.fs, args.notch_width, args.notch_freq, 5., 3, 'butter',
                                                                               args.singled_out_filtered[cnt, :])

            #show_signal(S, args)
            #emd = EMD()
            #S = args.singled_out[cnt, :]
            #emd.emd(S)
            #imfs, res = emd.get_imfs_and_residue()
            #vis = Visualisation()
            #t = np.arange(0, S.size/args.fs, 1/args.fs)
            #vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
            #vis.plot_instant_freq(t, imfs=imfs)

            #vis.show()

            #args.singled_out_filtered_notched[cnt, :] = imfs[0]

            if args.show_midplots:
                print('notched')
                args.temp_add = 'notched'
                show_signal(args.singled_out_filtered[cnt, :], args)
                magnospec(args.singled_out_filtered_notched[cnt, :], args.fs)
    else:
        args.singled_out_filtered = args.singled_out
        args.singled_out_filtered_notched = args.singled_out

    # low_band = 0.25
    # high_band = 100
    # args.singled_out_filtered[cnt, :] = butter_bandpass(args.singled_out[cnt, :], low_band, high_band, args.fs, order)

    return args

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


def butter(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_filter(data, cutoff, fs, order=5):
    b, a = butter(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def implement_notch_filter(fs, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter, lfilter
    nyq = fs/2.0
    low = freq - band/2.0
    high = freq + band/2.0
    low = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data


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


def get_stats(args):

    sns.set(color_codes=True)
    args.der = np.diff(args.times)
    # args.der = args.der[args.der < 5000]
    # args.der = np.delete(args.der, args.der > 3)
    args.ave_der = np.mean(args.der)
    args.std_der = np.std(args.der)
    sns.distplot(args.der, hist=False, rug=True)
    plt.show()
    plt.savefig(os.path.join(args.output_dir, 'soa_distribution.png'))
    plt.close()

    return args


def reref_all(selected_channels):
    meanref = np.mean(selected_channels, 0)
    return meanref


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def ideally():

    # reref
    # filter lp
    # notch
    # ...

    return 0


# temp = stats.zscore(ave_y2_500)
# print(temp)
# print(max(temp))
# print(np.argmax(temp))
# print(len(temp))
# show_signal(ave_y2_500)
# show_signal(temp)
