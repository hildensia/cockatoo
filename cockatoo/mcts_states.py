from enum import Enum
from copy import deepcopy
import numpy as np
from joint_dependency.inference import exp_neg_entropy
from scipy.stats import entropy

from cockatoo.belief import JointDependencyBelief


class JointDependencyAction(Enum):
    explore = 0
    exploit = 1


class JointDependencyState(object):
    def __init__(self, belief):
        self.performed = {}
        self.belief = belief
        self.objective_fnc = exp_neg_entropy
        self.n_samples = 1000

    def perform(self, action):
        sim_action = self._get_best_action(action)
        experience = self.belief.simulate(sim_action)

        experiences = deepcopy(self.belief.experiences)
        experiences.append(experience)

        belief = JointDependencyBelief(experiences)
        return JointDependencyState(belief)

    def _get_best_action(self, action):
        try:
            return self.performed[action]
        except KeyError:
            if action == JointDependencyAction.explore:
                self.performed[action] = \
                    self.belief.get_best_explore_action(self.n_samples)
            elif action == JointDependencyAction.exploit:
                self.performed[action] = \
                    self.belief.get_best_exploit_action(self.n_samples)
            return self.performed[action]

    def reward(self, parent, _):
        open = 0
        n = 1000
        # estimate the probability of all open
        # (TODO: can we do it bayesian?)
        for _ in range(n):
            locking = self.belief.sample_locking(self.belief.pos)
            if not any(locking):
                open += 1
        reward = 100 * (open/n)
        return reward + entropy(parent.belief.posteriors, self.belief.posteriors)


