from scipy.io import loadmat

def readchannels():

    filename = 'data/1_20190919_010308.mat'
    data = loadmat(filename)

    channels = data['a1_20190919_010308mff1']
    fs = data['EEGSamplingRate'][0][0]
    stims = data['evt_ECI_TCPIP_55513']

    # debug_stop = 1

    return channels, fs, stims