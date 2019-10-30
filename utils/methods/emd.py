import numpy as np


def hes(args):

    # segment
    for cnt_ch in range(args.singled_out_filtered_notched.shape[0]):
        y2 = args.singled_out_filtered_notched[cnt_ch, :]
        # show_insta_phase(insta_phase_norm)

        wind_ = np.zeros([len(args.times), args.win_l + args.win_r])
        for cnti, i in enumerate(args.times):
            i1 = int(i)
            wind_[cnti, :] = y2[i1 - args.win_l: i1 + args.win_r]  # faster


        args.fr_res = 500
        sta = np.zeros([args.len_uc, 100,  args.fr_res, args.win_l + args.win_r])

        for i, uclass in enumerate(args.uc):
            ind = (args.uc_ind == i)
            temp = wind_[ind, :]
            for cnti in range(temp.shape[0]):
                test = stft(temp[cnti, :])  # calc hist wind_[ind, :]
                sta[i, cnti, :, :] = test

            # step = np.abs(np.mean(phase_reset_wind[ind, :]))
            # mean_step = np.mean()

            sta_mean[i, :, :] = np.abs(np.mean(sta[i, :, :, :], 0))
            sta_std[i, :, :] = np.abs(np.std(sta[ind, :, :, :], 0))

    #         testimage = np.squeeze(hist_wind[i, :, :])
    #         # fig, ax = plt.subplots(nrows=1, ncols=1)
    #         plt.imshow(testimage)  # , aspect='auto'
    #         plt.title(np.max(testimage))
    #         #ax.set_adjustable('box-forced')
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
