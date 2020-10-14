from baselines.common import plot_util as pu

import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


results_per = pu.load_results('./final-logs/breakout-per/BreakoutNoFrameskip-v0-20201009-per')
results_noper = pu.load_results('./final-logs/breakout-noper/BreakoutNoFrameskip-noper-v0-20201009')
results_closest = pu.load_results('./final-logs/breakout-closest/BreakoutNoFrameskip-v0-20201006')
results_pdd = pu.load_results('/Users/khanh/utas/thesis/openai-baseline/baselines/final-logs/breakout-pddper/logs/BreakoutNoFrameskip-v0-20201009-pddper')
plt.xlabel('episodes')
plt.ylabel('rewards')
r = results_noper[0]
plt.plot(running_mean(r.monitor.r.values, 100), label='dqn')

r_per = results_per[0]
plt.plot(running_mean(r_per.monitor.r.values, 100), label='per')

r_uniform = results_closest[0]
plt.plot(running_mean(r_uniform.monitor.r.values, 100), label='uniform')

r_pddper = results_pdd[0]
plt.plot(running_mean(r_pddper.monitor.r.values, 100), label='pdd-dqn')
plt.legend()
plt.savefig('breakout-result.png', bbox_inches='tight')
