from joint_dependency.inference import (model_posterior, prob_locked,
                                        exp_cross_entropy)
import numpy as np
from .utils import rand_max
from scipy.optimize import minimize


def ass_bool(obj):
    assert isinstance(obj, bool) or isinstance(obj, np.bool_)


class JointDependencyBelief(object):
    def __init__(self, pos, experiences, belief=None):
        self.experiences = experiences
        self.pos = np.asarray(pos)

        if belief is not None:
            self.p_same = belief.p_same
            self.alpha_prior = belief.alpha_prior
            self.model_prior = belief.model_prior

        self._posteriors = None

    @property
    def posteriors(self):
        if self._posteriors is not None:
            return self._posteriors
        posteriors = []
        for i, _ in enumerate(self.pos):
            posteriors.append(model_posterior(self.experiences[i], self.p_same,
                                              self.alpha_prior,
                                              self.model_prior[i]))
        self._posteriors = posteriors
        return posteriors

    def sample_locking(self, pos):
        pos = np.asarray(pos)
        locking = np.ndarray(pos.shape, dtype=np.bool)
        for joint_idx, joint_exps in enumerate(self.experiences):
            pl = prob_locked(joint_exps, pos, self.p_same, self.alpha_prior,
                             None, self.posteriors[joint_idx])
            # pl is a dirichlet distribution object. Draw a sample and sample
            # from that
            locking[joint_idx] = np.random.uniform() < pl.rvs()[0][0]
        return locking

    def prob_locking(self, pos):
        pos = np.asarray(pos)
        pl = []
        for joint_idx, joint_exps in enumerate(self.experiences):
            pl.append(prob_locked(joint_exps, pos, self.p_same, self.alpha_prior,
                                  None, self.posteriors[joint_idx]))
        return pl

    def simulate(self, action, sim_joint, locking):
        #locking = self.sample_locking(self.pos)

        new_pos = np.array(self.pos)

        for joint, pos in enumerate(action):
            if not locking[joint]:
                new_pos[joint] = pos

        locking = self.sample_locking(new_pos)
        return locking, new_pos

    def sample_joint_states(self, n, locking):
        samples = []
        #print("pos: {}".format(self.pos))
        #print("locking: {}".format(locking))
        for i in range(n):
            pos = self.pos

            #locked = self.sample_locking(self.pos)

            for j, joint in enumerate(self.pos):
                if not locking[j]:
                    pos[j] = np.random.randint(0, 180)  # TODO: add
                                                        # proper limits

            # print("npos: {}".format(pos))
            samples.append(pos)
        return samples

    def get_best_explore_action(self, locking):
        key = lambda pos: np.sum([exp_cross_entropy(self.experiences[joint],
                                                    pos,
                                                    self.p_same,
                                                    self.alpha_prior,
                                                    self.model_prior[joint])
                                  for joint in range(self.pos.shape[0])])

        return self._optimize(key, locking)

        # samples = self.sample_joint_states(n_samples, locking)
        # return rand_max(samples,
        #

    def get_best_unlock_action(self, n_samples, joint, locking):
        key = lambda pos: ((prob_locked(self.experiences[joint],
                                        pos,
                                        self.p_same,
                                        self.alpha_prior,
                                        self.model_prior[joint])).mean())[1]

        return self._optimize(key, locking)

        # samples = self.sample_joint_states(n_samples, locking)
        # return rand_max(samples,
        #                 key=lambda pos: ((prob_locked(self.experiences[joint],
        #                                             pos,
        #                                             self.p_same,
        #                                             self.alpha_prior,
        #                                             self.model_prior[joint])).mean())[1])

    def get_best_lock_action(self, n_samples, joint, locking):
        key = lambda pos: ((prob_locked(self.experiences[joint],
                                        pos,
                                        self.p_same,
                                        self.alpha_prior,
                                        self.model_prior[joint])).mean())[0]
        return self._optimize(key, locking)

    def _sample_unlocked(self, n, locking):
        samples = []

        i = 0

        if all(locking):
            samples.append(self.pos)
            return samples

        while i < n:
            i += 1
            pos = np.copy(self.pos)

            j = np.random.choice(np.where(-np.array(locking))[0])
            pos[j] = np.random.randint(0, 180)

            p_same = np.prod([self.p_same[i][self.pos[i]][pos[i]]
                              for i in range(pos.shape[0])])

            # if np.random.uniform() > p_same:
            #     samples.append(pos)
            # elif np.random.uniform() > 0.95:
            samples.append(pos)
            # else:
            #     # print(pos, p_same)
            #     i -= 1

        return samples

    def _optimize(self, key, locking, verbose=0):
        samples = self._sample_unlocked(100, locking)
        return rand_max(samples, key=key)
