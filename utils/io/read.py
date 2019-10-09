from scipy.io import loadmat

def readchannels():


    if(1):
        filename = 'data/Tina 10-8-19 round 2 1_20191008_062255'
        data = loadmat(filename)
        channels = data['Tina10819round21_20191008_062255mff2']  # data['a1_20190919_010308mff1']
    #        filename = 'data/2_20190919_010845.mat'
#        data = loadmat(filename)
#        channels = data['a2_20190919_010845mff2']  # data['a1_20190919_010308mff1']
    else:
        filename = 'data/1_20190919_010308.mat'
        data = loadmat(filename)
        channels = data['a1_20190919_010308mff2']

    fs = data['EEGSamplingRate'][0][0]
    stims = data['evt_ECI_TCPIP_55513']

    return channels, fs, stims
