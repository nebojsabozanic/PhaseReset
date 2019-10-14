import numpy as np
from utils.disp.showphases import show_csignals


def n1p1(signal, times, wind_l, wind_r):

    wind_ = np.zeros([len(times), wind_l + wind_r])
    for cnti, i in enumerate(times):

        i1 = i[0].astype(int)
        wind_[cnti, :] = signal[i1[0] - wind_l : i1[0] + wind_r]

    ave_wind_1 = np.mean(wind_[0:100, :], 0)
    std_wind_1 = np.std(wind_[0:100, :], 0)
    ave_wind_2 = np.mean(wind_[100:200, :], 0)
    std_wind_2 = np.std(wind_[100:200, :], 0)

    return ave_wind_1, std_wind_1, ave_wind_2, std_wind_2


def n1p1c(signal, times, wind_l, wind_r, uc, uc_ind, len_uc):
    wind_ = np.zeros([len(times), wind_l + wind_r])
    for cnti, i in enumerate(times):

        i1 = i[0].astype(int)
        wind_[cnti, :] = signal[i1[0] - wind_l : i1[0] + wind_r] # faster

    #print(len_uc)
    ave_wind = np.zeros([len_uc, wind_l + wind_r])
    std_wind = np.zeros([len_uc, wind_l + wind_r])

    for i, uclass in enumerate(uc):
        ind = (uc_ind == i)
        # print(i)
        # print(ind)
        ave_wind[i, :] = np.mean(wind_[ind, :], 0)
        std_wind[i, :] = np.std(wind_[ind, :], 0)

    return ave_wind, std_wind


def reref1(single, all):



    return 0


def rerefAll(all):

    meanref = np.mean(all, 0)

    return meanref


def getN1P1(args):

    for cnt in range(args.channels.shape[0]):
        signal = args.singled_out_filtered_notched[cnt,:]
        wind_ = np.zeros([len(args.times), args.win_l + args.win_r])
        for cnti, i in enumerate(args.times):
            i1 = i[0].astype(int)
            wind_[cnti, :] = signal[i1[0] - args.win_l: i1[0] + args.win_r]  # faster

        # print(len_uc)
        ave_wind = np.zeros([args.len_uc, args.win_l + args.win_r])
        std_wind = np.zeros([args.len_uc, args.win_l + args.win_r])

        for i, uclass in enumerate(args.uc):
            ind = (args.uc_ind == i)
            # print(i)
            # print(ind)
            ave_wind[i, :] = np.mean(wind_[ind, :], 0)
            std_wind[i, :] = np.std(wind_[ind, :], 0)

        show_csignals(ave_wind, args.output_dir, cnt)

    return 0

# ave_y2_500, std_y2_500, ave_y2_1000, std_y2_1000 = n1p1(y2, stims, 400, 2000)
# ave_y2, std_y2 = n1p1c(y2, stims, 100, 1000, uc, uc_ind, len_uc)
# print(np.size(ave_y2))

# show_2signals(ave_y2_500, ave_y2_1000, output_dir, cnt)
# show_csignals(ave_y2, output_dir, cnt)
