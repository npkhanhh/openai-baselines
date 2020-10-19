from baselines.common import plot_util as pu

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def load_result(path):
    result = pu.load_results(path)
    result = result[0]
    return result.monitor.r.values


final_log_dir = '/Users/khanh/logs/final-logs'


def get_avgs(env):
    methods = ['per', 'noper', 'uniform']
    results = []
    for m in methods:
        a = []
        for i in range(1, 4):
            trial_path = os.path.join(final_log_dir, env, m, 'trial'+str(i))
            trial_result = load_result(trial_path)
            a.append(trial_result)
        l = min(len(a[0]), len(a[1]), len(a[2]))
        a[0] = a[0][:l]
        a[1] = a[1][:l]
        a[2] = a[2][:l]
        avgs = np.mean(a, axis=0)
        results.append(avgs)
    return results

env='acrobot'
per, dqn, uniform = get_avgs(env.lower())
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.plot(running_mean(dqn, 100), label='dqn')
plt.plot(running_mean(per, 100), label='per')
plt.plot(running_mean(uniform, 100), label='uniform')
plt.title('100-episode moving average over 3 trials')
plt.legend()
# plt.show()
plt.savefig('acrobot-result.png', bbox_inches='tight')
