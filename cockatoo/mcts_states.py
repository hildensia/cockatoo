from enum import Enum
import numpy as np
from joint_dependency.inference import exp_neg_entropy

from cockatoo.belief import JointDependencyBelief


class JointDependencyAction(Enum):
    explore = 0
    exploit = 1


class JointDependencyState(object):
    def __init__(self, belief, simulator):
        self.performed = {JointDependencyAction.exploit: False,
                          JointDependencyAction.explore: False}
        self.belief = belief
        self.simulator = simulator
        self.objective_fnc = exp_neg_entropy

    def perform(self, action):
        sim_action = self._get_best_action(action)
        belief_sample = self.belief.sample()
        sim_state = self.simulator.perform(belief_sample, sim_action)
        belief = JointDependencyBelief(sim_state)
        return JointDependencyState(belief, self.simulator)

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
            pos = np.ndarray((len(self.simulator.world.joints),))
            for j, joint in enumerate(self.simulator.world.joints):
                if self.belief.locked_states[j] == 1:
                    pos[j] = int(joint.get_q())
                else:
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
        # compute exp_neg_entropy
        pass

