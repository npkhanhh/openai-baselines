from baselines.common import plot_util as pu

import matplotlib.pyplot as plt
import numpy as np
import subprocess


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


line_noper = subprocess.check_output(
    ['tail', '-1', '/Users/khanh/utas/thesis/openai-baseline/baselines/baselines/td_error_CartPole-v1_noper.txt']).rstrip()
line_uniform = subprocess.check_output(
    ['tail', '-1', '/Users/khanh/utas/thesis/openai-baseline/baselines/baselines/td_error_CartPole-v1_uniform.txt']).rstrip()

a_noper = sorted(map(float, line_noper.split()))
print(a_noper)
a_uniform = list(map(float, line_uniform.split()))
# plt.hist(a_uniform_noper)
print(min(a_noper), max(a_noper))
plt.hist(a_noper, bins=np.arange(min(a_noper[:-1]), max(a_noper[:-1]) + 0.01, 0.01))
# plt.hist(a_uniform, bins=np.arange(min(a_uniform[:-1]), max(a_uniform[:-1]) + 0.01, 0.01))

plt.show()
