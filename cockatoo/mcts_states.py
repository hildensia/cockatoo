from enum import Enum
from copy import deepcopy
import numpy as np
from joint_dependency.inference import exp_neg_entropy

from cockatoo.belief import JointDependencyBelief


class JointDependencyAction(Enum):
    explore = 0
    exploit = 1


class JointDependencyState(object):
    def __init__(self, belief):
        self.performed = {}
        self.belief = belief
        self.objective_fnc = exp_neg_entropy

    def perform(self, action):
        sim_action = self._get_best_action(action)
        experience = self.belief.simulate(sim_action)

        experiences = deepcopy(self.belief.experiences)
        experiences.append(experience)

        belief = JointDependencyBelief(experiences)
        return JointDependencyState(belief)

    def _get_best_action(self, action):
        if action == JointDependencyAction.explore:
            return self._get_best_explore_action()
        elif action == JointDependencyAction.exploit:
            return  self._get_best_exploit_action()

    def _get_best_explore_action(self):
        max_pos = None
        _max = - np.inf
        max_joint = None
        for i in range(self.N_samples):
            pos = self.belief.pos

            segments, dependency = self.belief.sample()
            locked = self.belief.sample_locking(self.belief.pos, segments,
                                                dependency)

            for j, joint in enumerate(self.belief.pos):
                if not locked[j]:
                    pos[j] = np.random.randint(joint.min_limit,
                                               joint.max_limit)

            joint = np.random.randint(0, len(self.simulator.world.joints))
            value = self.objective_fnc(self.belief.experiences[joint],
                                       pos,
                                       self.belief.p_same,
                                       self.belief.alpha_prior,
                                       self.belief.model_prior[joint])

            if value > _max:
                _max = value
                max_pos = pos
                max_joint = joint
        return max_pos, max_joint

    def _get_best_exploit_action(self):
        pass

    def reward(self):
        # TODO: compute exp_neg_entropy
        pass

