import numpy as np


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


def reref1(single, all):



    return 0


def rerefAll(all):

    meanref = np.mean(all, 0)

    return meanref
