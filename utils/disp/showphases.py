import matplotlib.pyplot as plt
# import os.path


def show_signal(signal):  # , t, output_dir, filename, axisname):
    plt.plot(signal)  # t,
    # plt.title('Original signal')
    # plt.ylabel('Amplitude')
    # plt.xlabel(axisname)
    # plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    return 0


def show_windows(signal, tk):

    win_l = 100
    win_r = 2000
    for i in tk:

        check0 = i[0].astype(int)
        check1 = check0-win_l
        check2 = check0+win_r
        f = plt.figure
        plt.plot(signal[check1[0]:check2[0]])  # t,
        plt.show()
        plt.waitforbuttonpress(0.1)
        plt.close('all')

    return 0


def showphases():

    return 0
