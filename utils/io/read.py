from scipy.io import loadmat


def readchannels(args):

    # 1 Tina Session 1
    # filename = 'data/Tina 10-8-19 round 2 1_20191008_062255'
    # channel_name = 'Tina10819round21_20191008_062255mff2'

    # 5. Nebo click no less
    filename = 'data/1_20191011_044342 nebo loud clicks.mat'
    channel_name = 'a1_20191011_044342neboloudclicksmff2'

    # 4. Nebo clickless
    # filename = 'data/1_20191011_043914 nebo soft clicks.mat'
    # channel_name = 'a1_20191011_043914nebosoftclicksmff2'

    # 6. Nebo 1s clickless
    # filename = 'data/1_20191011_044634 nebo 1s clickless.mat'
    # channel_name = 'a1_20191011_044634nebo1sclicklessmff2'

    # 2 Nebo session 2
    # filename = 'data/2_20190919_010845.mat'
    # channel_name = 'a2_20190919_010845mff2'

    # 3 Nebo session 1
    # filename = 'data/1_20190919_010308.mat'
    # channel_name = 'a1_20190919_010308mff2'

    args.filename = filename
    args.channel_name = channel_name # til this line omit

    data = loadmat(args.filename)
    args.channels = data[args.channel_name]

    args.fs = data['EEGSamplingRate'][0][0]
    args.stims = data['evt_ECI_TCPIP_55513']

    return args


def surro(args):

    return args
