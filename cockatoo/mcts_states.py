from __future__ import division, print_function
from enum import Enum
from copy import deepcopy
import numpy as np
from joint_dependency.inference import exp_neg_entropy
from scipy.stats import entropy

from cockatoo.belief import JointDependencyBelief


def ass_bool(obj):
    assert isinstance(obj, bool) or isinstance(obj, np.bool_)


class JointDependencyActionType(Enum):

    explore = 0
    lock = 1
    unlock = 2


class JointDependencyAction(object):
    def __init__(self, type, joint_idx):
        self.type = type
        self.joint_idx = joint_idx

    def __eq__(self, other):
        return self.joint_idx == other.joint_idx and self.type == other.type

    def __hash__(self):
        return 100 * self.type.__hash__() + self.joint_idx

    def __str__(self):
        return self.type.name + " joint " + str(self.joint_idx)

    def __repr__(self):
        return self.__str__()


class JointDependencyState(object):
    def __init__(self, belief, locking, simulator, options):
        self.options = options
        self.performed = {}
        self.belief = belief
        self.locking = locking
        self.objective_fnc = exp_neg_entropy
        self.n_samples = 1000
        self.actions = [JointDependencyAction(type, joint)
                        for type in JointDependencyActionType
                        for joint in range(belief.pos.shape[0])]
        self.simulator = simulator

    def perform(self, action):
        sim_action = self.get_best_low_level_action(action)
        action.move_to = deepcopy(sim_action)
        locking, new_pos = self.belief.simulate(sim_action, action.joint_idx,
                                                self.locking)

        experiences = deepcopy(self.belief.experiences)
        for idx, l in enumerate(locking):
            experiences[idx].append({"data": new_pos, "value": 1 if l else 0})

        belief = JointDependencyBelief(new_pos, experiences)
        return JointDependencyState(belief, locking, self.simulator,
                                    self.options)

    def real_world_perform(self, action):
        experiences = deepcopy(self.belief.experiences)
        print(experiences)

        self.simulator.action_machine.run_action(
            self.get_best_low_level_action(action)
        )

        q = [joint.get_q() for joint in self.simulator.world.joints]
        locking = [j.locked
                   for j in self.simulator.world.joints]
        for idx, joint in enumerate(self.simulator.world.joints):
            exp = {"data": q, "value": 1 if joint.locked else 0}
            experiences[idx].append(exp)

        belief = JointDependencyBelief(q, experiences)
        return JointDependencyState(belief, locking, self.simulator,
                                    self.options)

    def get_best_low_level_action(self, action):
        try:
            return self.performed[action]
        except KeyError:
            if action.type == JointDependencyActionType.explore:
                self.performed[action] = \
                    deepcopy(self.belief.get_best_explore_action(self.locking))
            elif action.type == JointDependencyActionType.unlock:
                self.performed[action] = \
                    deepcopy(self.belief.get_best_unlock_action(
                        action.joint_idx, self.locking)
                    )
            elif action.type == JointDependencyActionType.lock:
                self.performed[action] = \
                    deepcopy(self.belief.get_best_lock_action(
                        action.joint_idx, self.locking)
                    )
            return self.performed[action]

    def reward(self, parent, _):
        reward = -1

        if self.options.goal_reward and not self.locking[4]:
            reward = 100

        if self.options.intrinsic_motivation:
            reward += sum(entropy(np.array(parent.belief.posteriors).T,
                                  np.array(self.belief.posteriors).T))

        return reward

    def is_terminal(self):
        return False

    def __eq__(self, other):
        return (self.locking == other.locking).all()

    def __hash__(self):
        return int(np.sum([l*2**i for i, l in enumerate(self.locking)]))
