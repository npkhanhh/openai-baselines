import numpy as np
import random
import bisect
from matplotlib import pyplot as plt

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size, env_name):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._td_errors = []
        self._replace_td_errors = []
        self._new_td = []
        self._expected_total = 0
        self._current_total = 0
        self.t = 0
        self._env_name = env_name

    def _update_min_max(self, td_error):
        if self._min_td_error <= td_error <= self._max_td_error:
            return
        if td_error < self._min_td_error:
            self._min_td_error = td_error
        if td_error > self._max_td_error:
            self._max_td_error = td_error
        self._expected_total = (self._min_td_error + self._max_td_error)

    def _calculate_expected_total(self):
        self._expected_total = (self._td_errors[0] + self._td_errors[-1]) * (self._maxsize / 2)

    def _get_insert_pos(self, td_error):
        pos = bisect.bisect_left(self._td_errors, td_error)
        return pos

    def _get_delete_pos(self, td_error):
        del_value = td_error - (self._expected_total - self._current_total)
        pos = bisect.bisect_left(self._td_errors, del_value)
        if pos == 0:
            return 1
        if pos == len(self._td_errors):
            return -2
        before = self._td_errors[pos - 1]
        after = self._td_errors[pos]
        if after - td_error < td_error - before:
            return pos
        else:
            return pos - 1

    def swap(self, insert_pos, delete_pos, content, td_error):
        data = [td_error, content]
        self._current_total += td_error - self._td_errors[delete_pos]
        if delete_pos < insert_pos:
            insert_pos -= 1
        del self._td_errors[delete_pos]
        del self._storage[delete_pos]
        self._td_errors.insert(insert_pos, td_error)
        self._storage.insert(insert_pos, data)

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, td_error):
        data = [td_error, (obs_t, action, reward, obs_tp1, done)]
        idx = self._get_insert_pos(td_error)
        if len(self._storage) < self._maxsize:
            bisect.insort(self._td_errors, td_error)
            self._storage.insert(idx, data)
            self._current_total += td_error
        else:
            self.swap(self._get_insert_pos(td_error), self._get_delete_pos(td_error), data[1], td_error)
        self._calculate_expected_total()
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if self._next_idx == 0:
            with open('td_error_{}_uniform2.txt'.format(self._env_name), 'a') as file:
                file.write(' '.join(map(str, self._td_errors)) + '\n')

    def update(self, batch_idxs, td_error):
        for idx, error in zip(batch_idxs, td_error):
            old_error, data = self._storage[idx]
            self.swap(self._get_insert_pos(error), idx, data, error)
        self._calculate_expected_total()

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        data = self._storage[0][1]
        ob_dtype = data[0].dtype
        ac_dtype = data[1].dtype
        for i in idxes:
            data = self._storage[i][1]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t, dtype=ob_dtype), np.array(actions, dtype=ac_dtype), \
               np.array(rewards, dtype=np.float32), np.array(obses_tp1, dtype=ob_dtype), np.array(dones,
                                                                                                  dtype=np.float32)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return tuple(list(self._encode_sample(idxes)) + [idxes])


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, env_name):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, env_name)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
