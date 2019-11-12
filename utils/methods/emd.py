import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation
from utils.disp.showphases import show_signal, show_csignals  # showphases


def hes2(args):
    n = 10000
    t = np.arange(0, n/args.fs, 1/args.fs)
    S = args.singled_out[-1, 0:n]
    args.temp_add = 'emd_raw'
    show_signal(S, args)
    emd = EMD()
    emd.emd(S)
    imfs, res = emd.get_imfs_and_residue()

    # In general:
    # components = EEMD()(S)
    # imfs, res = components[:-1], components[-1]

    vis = Visualisation()
    vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    vis.plot_instant_freq(t, imfs=imfs)

    vis.show()
    return 0

def hes(args):
    # fs = 10e3
    # N = 1e5
    # amp = 2 * np.sqrt(2)
    # noise_power = 0.01 * fs / 2
    # time = np.arange(N) / float(fs)
    # mod = 500 * np.cos(2 * np.pi * 0.25 * time)
    # carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
    # noise = np.random.normal(scale=np.sqrt(noise_power), size = time.shape)
    # noise *= np.exp(-time / 5)
    # x = carrier + noise
    #
    # f, t, Zxx = signal.stft(x, fs, nperseg=10)
    # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()


    # segment
    for cnt_ch in range(args.singled_out_filtered_notched.shape[0]):
        y2 = args.singled_out_filtered_notched[cnt_ch, :]
        # show_insta_phase(insta_phase_norm)

        wind_ = np.zeros([len(args.times), args.win_l + args.win_r])
        for cnti, i in enumerate(args.times):
            i1 = int(i)
            wind_[cnti, :] = y2[i1 - args.win_l: i1 + args.win_r]  # faster


        args.fr_res = 500
        sta = np.zeros([args.len_uc, 100, args.fr_res, args.fr_res, args.win_l + args.win_r])

        for i, uclass in enumerate(args.uc):
            ind = (args.uc_ind == i)
            temp = wind_[ind, :]
            for cnti in range(temp.shape[0]):
                what = temp[cnti, :]
                f, t, test = signal.stft(temp[cnti, :], args.fs, nperseg=100)  # calc hist wind_[ind, :]
                if cnti == 0:
                    stest = test
                else:
                    stest += test
                # f, t, Zxx = signal.stft(x, fs, nperseg=10)
                # emd = EMD()
                # IMFs = emd(what)

                # plt.plot(np.abs(signal.hilbert(IMFs[1, :])))
                # plt.pcolormesh(t, f, np.abs(test))
                # plt.title('STFT Magnitude')
                # plt.ylabel('Frequency [Hz]')
                # plt.xlabel('Time [sec]')
                # plt.show()
                # plt.waitforbuttonpress(0.1)
                # plt.close('all')

            plt.pcolormesh(t, f, np.abs(test))
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
            plt.waitforbuttonpress(0.1)
            plt.close('all')

                # sta[i, cnti, :, :] = np.abs(test)

            # step = np.abs(np.mean(phase_reset_wind[ind, :]))
            # mean_step = np.mean()

            #sta_mean[i, :, :] = np.abs(np.mean(sta[i, :, :, :], 0))
            #sta_std[i, :, :] = np.abs(np.std(sta[ind, :, :, :], 0))

    #         testimage = np.squeeze(hist_wind[i, :, :])
    #         # fig, ax = plt.subplots(nrows=1, ncols=1)
    #         plt.imshow(testimage)  # , aspect='auto'  `   
    #         plt.title(np.max(testimage))
    #         # ax.set_adjustable('box-forced')
    #         filename = 'histophases' + str(cnt_ch) + 'ch' + str(i) + 'cl' + '.png'
    #         plt.savefig(os.path.join(args.output_dir, filename), bbox_inches = 'tight', pad_inches = 0)
    #         plt.close()
    #         # plt.show()
    #         # plt.waitforbuttonpress(0.1)
    #         # save_ call show (methods>disp)
    #
    #     show_csignals(phase_reset_mean, phase_reset_std, args.output_dir, cnt_ch, 'phase_reset_indx')
    #
    #
    # # imfs = emd(args.)
    # # wavelet
    # # stft
    #
    # # plot hes
    return 0


def stft(signal):

    # short time fourier

    # wavelet

    # emd

    return 0
