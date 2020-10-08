from baselines.common import plot_util as pu

import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


results_per = pu.load_results('baseline-log/logs/mountaincar-v0')
results_1e5 = pu.load_results('baseline-log/logs/mountaincar-v0-1e5')
results_noper = pu.load_results('baseline-log/logs/mountaincar-v0-noper')

fig, axs = plt.subplots(1, 3, figsize=(25, 12), dpi=100, constrained_layout=True)

r = results_1e5[0]
axs[0].plot(np.cumsum(r.monitor.l), r.monitor.r)
axs[0].set_title('1e5 steps - per')
# axs[0].plot(np.cumsum(r.monitor.l), running_mean(r.monitor.r, 100))

r = results_per[0]
axs[1].plot(np.cumsum(r.monitor.l), r.monitor.r)
axs[1].set_title('1e6 steps - per')
# axs[1].plot(running_mean(r.monitor.r.tolist(), 100))
r = results_noper[0]
axs[2].plot(np.cumsum(r.monitor.l), r.monitor.r)
axs[2].set_title('1e6 steps - noper')
# axs[2].plot(running_mean(r.monitor.r.tolist(), 100))
plt.savefig('MountainCar-Baseline.jpg')
# plt.show()
