from joint_dependency.inference import (model_posterior, prob_locked,
                                        exp_cross_entropy)
import numpy as np
from .utils import rand_max_kv


def ass_bool(obj):
    assert isinstance(obj, bool) or isinstance(obj, np.bool_)


def _get_best_explore_action(tup):
    (belief, pos) = tup
    return np.sum([exp_cross_entropy(belief.experiences[joint],
                                     pos,
                                     JointDependencyBelief.p_same,
                                     JointDependencyBelief.alpha_prior,
                                     JointDependencyBelief.model_prior[joint],
                                     model_post=belief.posteriors[joint])
                   for joint in range(belief.pos.shape[0])])


def _get_best_lock_action(tup):
    (belief, joint, pos) = tup
    return ((prob_locked(belief.experiences[joint],
                         pos,
                         JointDependencyBelief.p_same,
                         JointDependencyBelief.alpha_prior,
                         JointDependencyBelief.model_prior[joint])).mean())[0]


def _get_best_unlock_action(tup):
    (belief, joint, pos) = tup
    return ((prob_locked(belief.experiences[joint],
                         pos,
                         JointDependencyBelief.p_same,
                         JointDependencyBelief.alpha_prior,
                         JointDependencyBelief.model_prior[joint])).mean())[1]


class JointDependencyBelief(object):
    pool = None
    alpha_prior = None
    model_prior = None
    p_same = None

    def __init__(self, pos, experiences):
        self.experiences = experiences
        self.pos = np.asarray(pos)
        self._posteriors = None

    @property
    def posteriors(self):
        if self._posteriors is not None:
            return self._posteriors
        posteriors = []
        for i, _ in enumerate(self.pos):
            posteriors.append(model_posterior(self.experiences[i],
                                              JointDependencyBelief.p_same,
                                              JointDependencyBelief.alpha_prior,
                                              JointDependencyBelief.model_prior[i]))
        self._posteriors = posteriors
        return posteriors

    def sample_locking(self, pos):
        pos = np.asarray(pos)
        locking = np.ndarray(pos.shape, dtype=np.bool)
        for joint_idx, joint_exps in enumerate(self.experiences):
            pl = prob_locked(joint_exps, pos, JointDependencyBelief.p_same,
                             JointDependencyBelief.alpha_prior,
                             None, self.posteriors[joint_idx])
            # pl is a dirichlet distribution object. Draw a sample and sample
            # from that
            locking[joint_idx] = np.random.uniform() < pl.rvs()[0][0]
        return locking

    def prob_locking(self, pos):
        pos = np.asarray(pos)
        pl = []
        for joint_idx, joint_exps in enumerate(self.experiences):
            pl.append(prob_locked(joint_exps, pos,
                                  JointDependencyBelief.p_same,
                                  JointDependencyBelief.alpha_prior,
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
        n = np.where(-np.array(locking))[0].shape[0]
        # print("n = {}".format(n))
        samples = self._sample_unlocked(n*90, locking)

        data = zip([self]*len(samples), samples)
        values = JointDependencyBelief.pool.map(_get_best_explore_action, data)

        kv = zip(samples, values)

        return rand_max_kv(kv)

    def get_best_unlock_action(self, joint, locking):
        n = np.where(-np.array(locking))[0].shape[0]
        # print("n = {}".format(n))
        samples = self._sample_unlocked(n*90, locking)

        data = zip([self]*len(samples), [joint]*len(samples), samples)
        values = JointDependencyBelief.pool.map(_get_best_unlock_action, data)

        kv = zip(samples, values)

        return rand_max_kv(kv)

    def get_best_lock_action(self, joint, locking):
        n = np.where(-np.array(locking))[0].shape[0]
        # print("n = {}".format(n))
        samples = self._sample_unlocked(n*90, locking)
        data = zip([self]*len(samples), [joint]*len(samples), samples)
        values = JointDependencyBelief.pool.map(_get_best_lock_action, data)

        kv = zip(samples, values)
        return rand_max_kv(kv)

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

            p_same = np.prod([JointDependencyBelief.p_same[k][self.pos[k]][pos[k]]
                              for k in range(pos.shape[0])])

            if np.random.uniform() > p_same:
                samples.append(pos)
            elif np.random.uniform() > 0.95:
                samples.append(pos)
            else:
                i -= 1

        return samples

