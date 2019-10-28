import numpy as np
from utils.disp.showphases import show_csignals  # , show_signal, magnospec


def get_erps(args):

    # args.ave_wind_all = np.zeros([args.channels.shape[0], args.len_uc, args.win_l + args.win_r])
    # args.std_wind_all = np.zeros([args.channels.shape[0], args.len_uc, args.win_l + args.win_r])

    for cnt in range(args.channels.shape[0]):
        signal = args.singled_out_filtered_notched[cnt, :]
        # show_signal(signal)
        # magnospec(signal, args.fs)
        wind_ = np.zeros([len(args.times), args.win_l + args.win_r])
        for cnti, i in enumerate(args.times):
            baseline = np.mean(signal[i - args.win_l: i])
            wind_[cnti, :] = signal[i - args.win_l: i + args.win_r] - baseline # baseline, faster

        ave_wind = np.zeros([args.len_uc, args.win_l + args.win_r])
        std_wind = np.zeros([args.len_uc, args.win_l + args.win_r])

        for i, uclass in enumerate(args.uc):
            ind = (args.uc_ind == i)
            ave_wind[i, :] = np.mean(wind_[ind, :], 0)
            std_wind[i, :] = np.std(wind_[ind, :], 0)

        # args.ave_wind_all[cnt, :, :] = ave_wind
        # args.std_wind_all[cnt, :, :] = std_wind

        show_csignals(ave_wind, std_wind, args.output_dir, cnt, 'erps')

    return args
