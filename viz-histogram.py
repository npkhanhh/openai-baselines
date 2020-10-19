from baselines.common import plot_util as pu

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import subprocess

normal = subprocess.check_output(['tail', '-1', '/Users/khanh/utas/thesis/openai-baseline/baselines/baselines/td_error_CartPole-v1__noper_1602819950.4454458.txt'])
uniform = subprocess.check_output(['tail', '-1', '/Users/khanh/utas/thesis/openai-baseline/baselines/baselines/td_error_CartPole-v1_uniform2_1603092330.487781.txt'])

normal = list(map(float, normal.rstrip().split()))
uniform = list(map(float, uniform.rstrip().split()))
# plt.hist(normal)
# plt.show()
plt.hist(uniform)
plt.show()
