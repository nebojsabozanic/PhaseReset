import matplotlib.pyplot as plt

# import os.path
from utils.methods.fouriers import power_spectrum_fft
import numpy as np
# from scipy import signal
import os.path


def show_insta_phase(signal1):
    plt.plot(signal1, 'b*')  # t,
    # plt.title('Original signal')
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    # plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    return 0


def show_signal(signal1, args):
    plt.plot(signal1)  # t,
    tempname = args.experiment + args.temp_add
    plt.title(tempname)
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    plt.savefig(os.path.join(args.output_dir, tempname))
    plt.show()
    plt.waitforbuttonpress(0.1)
    plt.close('all')
    return 0


def show_phase_reset(signal1, output_dir):

    filename = 'phasereset.png'
    plt.plot(signal1)  # t,
    # plt.title('Original signal')
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    return 0


def show_csignals(signal1, output_dir, cnt):
    t = np.arange(0, signal1.shape[1])
    for cnt1 in range(0, signal1.shape[0]):
        filename = 'erps' + str(cnt) + 'ch' + str(cnt1) + 'cl' + '.png'
        plt.plot(t, signal1[cnt1, :])  # t,
        # plt.show()
        # plt.waitforbuttonpress(0.1)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    # plt.ylim(-6, 6)
    # plt.title('Original signal')
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    # plt.savefig(os.path.join(output_dir, filename))
    # plt.show()
    # plt.waitforbuttonpress(0.1)
    # plt.close()
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
    # plt.show()
    return 0


def show_fft(p1, xf):

    plt.plot(xf, p1)
    # plt.ylim(-200, 20)
    # plt.xlim(0, 200)
    plt.show()

    return 0


def showphases(step3):

    plt.plot(np.real(step3), np.imag(step3), 'o')
    plt.show()
    plt.waitforbuttonpress(0.1)
    plt.close('all')

    return 0


def magnospec(signal1, fs):

    p1, xf = power_spectrum_fft(signal1, fs)
    p1m = 10 * np.log10(p1 * p1)  # / max(p1)) doesn't work
    show_fft(p1m, xf)

    return 0
