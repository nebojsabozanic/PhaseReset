import numpy as np
from scipy import signal
from utils.methods.fouriers import power_spectrum_fft
from utils.disp.showphases import showFFT, show_signal
import seaborn as sns
import matplotlib.pyplot as plt


def magnospec(signal, fs):

    p1, xf = power_spectrum_fft(signal, fs)
    p1m = 20 * np.log10(p1 / max(p1))
    showFFT(p1m, xf)
    return 0


def proc(args):

    if args.if_data_large:
        args.channels = args.channels[:, :10000]

    if len(args.channels[0, :]) % 2:
        args.singled_out = args.channels[:, :-1]  # potentially to all    args.singled_out_filtered = args.channels;
    else:
        args.singled_out = args.channels
    args.singled_out_filtered = args.singled_out
    args.singled_out_filtered_notched = args.singled_out

    args.classes = args.stims[0, 2:]

    # mostly for debugging
    if_data_large = 0
    do_rereferencing = 0

    args.times = args.stims[1, 2:]

    if args.if_data_large:
        dur1 = args.dur - args.win_r
        ind = args.stims[1, 2:] < dur1
        args.times = args.times[ind]
        args.classes = args.classes[ind]

    args.uc, args.uc_ind = np.unique(args.classes, return_inverse=True)
    args.len_uc = len(args.uc)

    for cnt in range(args.channels.shape[0]):

        # args.singled_out = args.channels[cnt, :]

        # show_signal(args.singled_out)

        # move this before the loop
        if if_data_large:
            args.singled_out[cnt, :] = args.singled_out[0:dur]

        # move this before the loop, take care, once you calc, then in the loop you take it out
        # re - referencing change to spatial
        # czr = reref(cz, channels)
        if do_rereferencing:
            meanref = rerefAll(args.channels)
            args.singled_out[cnt, :] -= meanref
                # show_signal(singled_out)

        # magnospec
        # tf = calc_stft(args.singled_out)
        # show_tf(tf)
        # show_windows(cz, stims, fs)

        # # notc
        order = 6
        cutoff = 3.667
        args.singled_out_filtered[cnt, :] = butter_filter(args.singled_out[cnt, :], cutoff, args.fs, order) # add to the object, and always take the last (as in the cell)

        # magnospec(args.singled_out_filtered[cnt, :], args.fs)

        #single_channel = args.channels[cnt, :]

        if (0):
            args.f0 = 60.
            args.Q = 10.
            b, a = signal.iirnotch(2 * args.f0 / args.fs, args.Q)
            args.singled_out_filtered_notched[cnt, :] = signal.filtfilt(b, a, args.singled_out_filtered[cnt, :])
            # show_signal(y2)
            # magnospec(y2)
        else:
            # show_signal(y1)
            # no hard coded!!!
            args.singled_out_filtered_notched[cnt, :] = Implement_Notch_Filter(1000., 0.5, 60., 5., 3, 'butter',
                                                                               args.singled_out_filtered[cnt, :])
            args.singled_out_filtered_notched[cnt, :] = Implement_Notch_Filter(1000., 0.5, 120., 5., 3, 'butter',
                                                                               args.singled_out_filtered[cnt, :])
            args.singled_out_filtered_notched[cnt, :] = Implement_Notch_Filter(1000., 0.5, 180., 5., 3, 'butter',
                                                                               args.singled_out_filtered[cnt, :])
            args.singled_out_filtered_notched[cnt, :] = Implement_Notch_Filter(1000., 0.5, 240., 5., 3, 'butter',
                                                                               args.singled_out_filtered[cnt, :])
            args.singled_out_filtered_notched[cnt, :] = Implement_Notch_Filter(1000., 0.5, 300., 5., 3, 'butter',
                                                                               args.singled_out_filtered[cnt, :])
            args.singled_out_filtered_notched[cnt, :] = Implement_Notch_Filter(1000., 0.5, 360., 5., 3, 'butter',
                                                                               args.singled_out_filtered[cnt, :])
            args.singled_out_filtered_notched[cnt, :] = Implement_Notch_Filter(1000., 0.5, 420., 5., 3, 'butter',
                                                                               args.singled_out_filtered[cnt, :])
            args.singled_out_filtered_notched[cnt, :] = Implement_Notch_Filter(1000., 0.5, 480., 5., 3, 'butter',
                                                                               args.singled_out_filtered[cnt, :])


            # show_signal(y2)
            # magnospec(args.singled_out_filtered_notched[cnt, :], args.fs)

        p1, xf = power_spectrum_fft(args.singled_out_filtered_notched[cnt, :], args.fs)
        showFFT(p1, xf)
        stop = 1

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


def getStats(args):

    sns.set(color_codes=True)
    args.der = np.diff(args.times)
    args.ave_der = np.mean(args.der)
    args.std_der = np.std(args.der)
    sns.distplot(args.der, hist=False, rug=True)
    # plt.show()
    plt.savefig('soa_distribution.png')
    plt.close()
    stop = 1

    return args
# temp = stats.zscore(ave_y2_500)
# print(temp)
# print(max(temp))
# print(np.argmax(temp))
# print(len(temp))
# show_signal(ave_y2_500)
# show_signal(temp)
