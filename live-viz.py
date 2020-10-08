from baselines.common import plot_util as pu

import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


while True:
    results = pu.load_results('/Users/khanh/logs/mountaincar-v0-20200825-no-per-modified-reward-buffer-size-20000')


    r = results[0]
    plt.plot(r.monitor.r)
    # plt.savefig('plain-MountainCar-modified-reward.jpg')

    plt.pause(10)
