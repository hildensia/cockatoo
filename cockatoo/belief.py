from joint_dependency.inference import (model_posterior, prob_locked,
                                        exp_neg_entropy)
import numpy as np
from .utils import rand_max


class JointDependencyBelief(object):
    def __init__(self, pos, experiences):
        self.experiences = experiences
        self.pos = np.asarray(pos)

        self.p_same = [[], []]
        self.p_dependencies = [[], []]
        self.p_locking = [[], []]

        self.alpha_prior = 0
        self.model_prior = [0]

    @property
    def posteriors(self):
        posteriors = []
        for i, _ in enumerate(self.pos):
            posteriors.append(model_posterior(self.experiences[i], self.p_same,
                                              self.alpha_prior,
                                              self.model_prior[i]))
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

    def simulate(self, action, sim_joint):
        locking = self.sample_locking(self.pos)

        new_pos = self.pos

        for joint, pos in enumerate(action):
            if not locking[joint]:
                new_pos[joint] = pos

        locking = self.sample_locking(new_pos)
        return {"data": new_pos, "value": locking[sim_joint]}

    def sample_joint_states(self, n):
        samples = []
        for i in range(n):
            pos = self.pos

            locked = self.sample_locking(self.pos)

            for j, joint in enumerate(self.pos):
                if not locked[j]:
                    pos[j] = np.random.randint(0, 180)  # TODO: add
                                                        # proper limits

            joint = np.random.randint(0, self.pos.shape[0])
            samples.append((pos, joint))
        return samples

    def get_best_explore_action(self, n_samples):
        samples = self.sample_joint_states(n_samples)
        return rand_max(samples,
                        key=lambda x: exp_neg_entropy(self.experiences[x[1]],
                                                      x[0],
                                                      self.p_same,
                                                      self.alpha_prior,
                                                      self.model_prior[x[1]]))

    def get_best_unlock_action(self, n_samples):
        samples = self.sample_joint_states(n_samples)
        return rand_max(samples,
                        key=lambda x: -prob_locked(self.experiences[x[1]],
                                                   x[0],
                                                   self.p_same,
                                                   self.alpha_prior,
                                                   self.model_prior[x[1]]))

    def get_best_lock_action(self, n_samples):
        samples = self.sample_joint_states(n_samples)
        return rand_max(samples,
                        key=lambda x: prob_locked(self.experiences[x[1]],
                                                  x[0],
                                                  self.p_same,
                                                  self.alpha_prior,
                                                  self.model_prior[x[1]]))
