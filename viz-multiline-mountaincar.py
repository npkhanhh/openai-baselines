from baselines.common import plot_util as pu

import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


no_per = pu.load_results('/Users/khanh/logs/mountaincar-v0-20201025-noper-trial2')
no_per_2 = pu.load_results('/Users/khanh/logs/mountaincar-v0-20201025-noper-trial2-20k')
plt.xlabel('episodes')
plt.ylabel('rewards')
r = no_per[0]
plt.plot(running_mean(r.monitor.r.values, 100), label='dqn')

r_per = no_per_2[0]
plt.plot(running_mean(r_per.monitor.r.values, 100), label='per')

plt.legend()
plt.show()
# plt.savefig('acrobot-result.png', bbox_inches='tight')
