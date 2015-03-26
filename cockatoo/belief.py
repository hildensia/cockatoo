from joint_dependency.inference import (model_posterior, prob_locked)
import numpy as np


def calc_once(f):
    def wrapper(*args, **kwargs):
        if wrapper.result is None:
            wrapper.result = f(*args, **kwargs)
        return wrapper.result
    wrapper.result = None
    return wrapper


class JointDependencyBelief(object):
    def __init__(self, pos, experiences):
        self.experiences = experiences
        self.pos = pos

        self.p_same = [[], []]
        self.p_dependencies = [[], []]
        self.p_locking = [[], []]

        self.alpha_prior = 0
        self.model_prior = [0]

        self._posteriors = None

    @property
    @calc_once
    def posteriors(self):
        posteriors = []
        for i, _ in enumerate(self.pos):
            posteriors.append(model_posterior(self.experiences[i], self.p_same,
                                              self.alpha_prior,
                                              self.model_prior[i]))
        return posteriors

    def sample_locking(self, pos):
        locking = np.ndarray(pos.shape, dtype=np.bool)
        for joint_idx, joint_exps in enumerate(self.experiences):
            pl = prob_locked(joint_exps, pos, self.p_same, self.alpha_prior,
                             None, self.posteriors[joint_idx])
            locking[joint_idx] = np.random.uniform() < pl
        return locking

    def simulate(self, action, sim_joint):
        locking = self.sample_locking(self.pos)

        new_pos = self.pos

        for joint, pos in enumerate(action):
            if not locking[joint]:
                new_pos = pos

        locking = self.sample_locking(new_pos)
        return {"data": new_pos, "value": locking[sim_joint]}





