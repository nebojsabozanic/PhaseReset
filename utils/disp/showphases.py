# import matplotlib
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

# import os.path
from utils.methods.fouriers import power_spectrum_fft
import numpy as np
from scipy import signal
import os.path
import seaborn as sns


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def show_insta_phase(signal):
    plt.plot(signal, 'b*')  # t,
    # plt.title('Original signal')
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    # plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    return 0


def show_signal(signal):  # , t, output_dir, filename, axisname):
    plt.plot(signal)  # t,
    # plt.title('Original signal')
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    # plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    return 0


def show_phase_reset(signal, output_dir):  # , t, output_dir, filename, axisname):
    filename = 'phasereset.png'
    plt.plot(signal)  # t,
    # plt.title('Original signal')
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    return 0


def show_examples(args):

    channel_no = 255 # rand
    sns.set(style="darkgrid")

    # Load an example dataset with long-form data
    fmri = sns.load_dataset("fmri")

    # Plot the responses for different events and regions
    sns.lineplot(x="timepoint", y="signal",
                 hue="region", style="event",
                 data=fmri)

    plt.show()

    # Plot the responses for different events and regions
    sns.lineplot(x="timepoint", y="signal",
                 hue="region", style="event",
                 data=np.squeeze(args.ave_wind_all[channel_no, 1, :]))

    plt.show()

    show_signal(np.squeeze(args.ave_wind_all[channel_no, 1, :]))
    stimulus_no = 15 # rand
    # show random raw
    i = args.times[stimulus_no]
    check1 = int(i[0][0]) - args.win_l
    check2 = int(i[0][0]) + args.win_r
    wind = args.singled_out_filtered_notched[channel_no, check1 : check2]
    show_signal(wind)
    # show random filtered
    stop = 1
    # show random filtered-notch

    # p1, xf = power_spectrum_fft(args.singled_out_filtered_notched[cnt, :], args.fs)
    # showFFT(p1, xf)

    return 0


def show_csignals(signal, output_dir, cnt):  # , t, output_dir, filename, axisname):
    print(signal.shape[0])
    t = np.arange(0, signal.shape[1])
    for cnt1 in range(0, signal.shape[0]):
        filename = str(cnt) + 'ch' + str(cnt1) + 'cl' + '.png'
        plt.plot(t, signal[cnt1, :])  # t,
        #plt.show()
        #plt.waitforbuttonpress(0.1)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    #plt.ylim(-6, 6)
    # plt.title('Original signal')
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    #plt.savefig(os.path.join(output_dir, filename))
    #plt.show()
    #plt.waitforbuttonpress(0.1)
    #plt.close()
    return 0


def show_2signals(signal1, signal2, output_dir, cnt):  # , t, output_dir, filename, axisname):
    filename = str(cnt) + '.png'
    t = np.arange(0, len(signal1))
    plt.plot(t, signal1, t, signal2)  # t,
    plt.ylim(-6, 6)
    # plt.title('Original signal')
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    #plt.show()
    return 0


def show_windows(x, tk, fs):

    plot_bool = 1
    win_l = 100

    win_l = 0 # -100 -1000 # baseline test
    win_r = 1600 # 1500 #2000
    Zxx_all = 0
    counter = 0
    for i in tk:
        counter += 1
        print(counter)
        if i[0] < 10000:
            continue

        check0 = i[0].astype(int)
        check1 = check0-win_l
        check2 = check0+win_r
        # plt.plot(x[check1[0]:check2[0]])  # t,
        #plt.show()
        #plt.waitforbuttonpress(0.1)
        #plt.close('all')
        # P1, xf = calcFFT(x[check1[0]:check2[0]], fs)
        # showFFT(P1, xf)
        #plt.waitforbuttonpress(0.1)
        #plt.close('all')
        # f, t, Zxx = signal.stft(x[check1[0]:check2[0]], fs)
        x1 = x[check1[0]:check2[0]]
        # Filter the data, and plot both the original and filtered signals.
        order = 6
        cutoff = 3.667
        y1 = butter_lowpass_filter(x1, cutoff, fs, order)
        f0 = 60.
        Q = 30.
        b, a = signal.iirnotch(2*f0/fs, Q)
        y2 = signal.filtfilt(b, a, y1)
        #t = np.linspace(0, 2.1, 2100, endpoint=False)
        #plt.plot(t, x1, 'g-', t, y1, 'b-')  # t,

        if (0):
            plt.plot(y2)  # t,
            plt.show()
            plt.waitforbuttonpress(0.1)
            plt.close('all')
            P1, xf = calcFFT(y2, fs)
            showFFT(P1, xf)
            plt.waitforbuttonpress(0.1)
            plt.close('all')
            # N = len(x1)
        # amp = 2 * np.sqrt(2)
        # noise_power = 0.01 * fs / 2
        # time = np.arange(N) / float(fs)
        # mod = 500 * np.cos(2 * np.pi * 0.25 * time)
        # carrier = amp * np.sin(2 * np.pi * 3e1 * time + mod)
        # noise = np.random.normal(scale=np.sqrt(noise_power), size = time.shape)
        # noise *= np.exp(-time / 5)
        # x1 = carrier + noise
        f, t, Zxx = signal.stft(y2, fs, nperseg=70)
        #Zxx[0:1, :] = 0
        #Zxx[:, 0:1] = 0
        Zxx_all += np.exp(np.abs(Zxx))

        if plot_bool:
            showSTFT(f, t, Zxx)
            print(np.max(np.abs(Zxx)))
            plt.waitforbuttonpress(0.1)
            plt.close('all')

    showSTFT(f, t, np.log(Zxx_all))
    plt.waitforbuttonpress(0.1)
    plt.close('all')
    return 0


def showFFT(P1, xf):

    plt.plot(xf, P1)
    #plt.ylim(-200, 20)
   #  plt.xlim(0, 200)
    plt.show()

    return 0


def showSTFT(f, t, Zxx):

    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax = 20)  # , vmax=amp)
    plt.title('STFT')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    return 0


def showphases(step3):

    plt.plot(np.real(step3), np.imag(step3), 'o')
    plt.show()
    plt.waitforbuttonpress(0.1)
    plt.close('all')

    return 0

def magnospec(signal, fs):

    p1, xf = power_spectrum_fft(signal, fs)
    #p1m = 20 * np.log10(p1 / max(p1))
    showFFT(p1, xf)
    return 0
