from baselines.common import plot_util as pu

import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


no_per = pu.load_results('/Users/khanh/logs/mountaincar-v0-20201015-noper')
no_per_2 = pu.load_results('/Users/khanh/logs/mountaincar-v0-20201015-noper-trial2')
no_per_3 = pu.load_results('/Users/khanh/logs/mountaincar-v0-20201015-noper-trial3')
no_per_20k = pu.load_results('/Users/khanh/logs/mountaincar-v0-20201015-noper-trial2-20k')
no_per_2_20k = pu.load_results('/Users/khanh/logs/mountaincar-v0-20201015-noper-trial3-20k')
plt.xlabel('episodes')
plt.ylabel('rewards')
r = results_noper[0]
plt.plot(running_mean(r.monitor.r.values, 100), label='dqn')

r_per = results_per[0]
plt.plot(running_mean(r_per.monitor.r.values, 100), label='per')

r_uniform = results_closest[0]
plt.plot(running_mean(r_uniform.monitor.r.values, 100), label='uniform')
plt.legend()
plt.savefig('acrobot-result.png', bbox_inches='tight')
