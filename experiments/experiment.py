from cockatoo.mcts_states import JointDependencyBelief, JointDependencyState
from joint_dependency.inference import same_segment
from joint_dependency.simulation import (Joint, MultiLocker, Controller,
                                         ActionMachine, World)
import numpy as np
import mcts.graph
import mcts.mcts
import mcts.backups
import mcts.tree_policies
import mcts.default_policies

import cProfile

class Simulator(object):
    def __init__(self, noise):
        self.world = World([])

        #  master | slave | open | closed
        # --------+-------+------+--------
        #    0    |   1   | 160+ |  160-
        #    1    |   2   | 160+ |  160-
        #    2    |   3   | 160+ |  160-
        #    3    |   4   | 160+ |  160-

        closed = (0, 160)
        limits = (0, 180)

        states = [closed[0], closed[1]]
        dampings = [15, 200, 15]
        for _ in range(5):
            self.world.add_joint(Joint(states, dampings, limits=limits,
                                       noise=noise))

        for i in range(1, 5):
            MultiLocker(self.world, locker=self.world.joints[i-1],
                        locked=self.world.joints[i], locks=[closed])

        controllers = [Controller(self.world, j)
                       for j, _ in enumerate(self.world.joints)]
        self.action_machine = ActionMachine(self.world, controllers)


def main():
    n = 5
    experiences = [[]]*n
    print(experiences)
    for i in range(n):
        experiences[i] = [{"data": np.array([0]*n), "value": 1}]
    experiences[0][0]["value"] = 0

    print(experiences)

    belief = JointDependencyBelief([0] * n, experiences)
    print("Pos: {}".format(belief.pos))

    belief.alpha_prior = np.array([.1, .1])

    p_cp = np.array([.01] * 360)
    p_cp[160] = .99

    belief.p_same = [same_segment(p_cp)] * n

    independent_prior = .6

    # the model prior is proportional to 1/distance between the joints
    model_prior = np.array([[0 if x == y
                             else independent_prior if x == n
                             else 1/abs(x-y)
                             for x in range(n+1)]
                            for y in range(n)])

    # normalize
    model_prior[:, :-1] = ((model_prior.T[:-1, :] /
                            np.sum(model_prior[:, :-1], 1)).T *
                           (1-independent_prior))
    print(model_prior)
    belief.model_prior = model_prior
    noise = {'q': 10e-6, 'vel': 10e-6}
    state = JointDependencyState(belief, [False, True, True, True, True],
                                 Simulator(noise))

    print(belief.posteriors)

    root = mcts.graph.StateNode(None, state)
    search = mcts.mcts.MCTS(tree_policy=mcts.tree_policies.UCB1(c=10),
                            default_policy=mcts.default_policies.immediate_reward,
                            backup=mcts.backups.Bellman(gamma=.6))

    state_node = root
    for _ in range(100):
        action = search(state_node, n=10)
        state_node = state_node.children[action].sample_state(real_world=True)
        state_node.parent = None
        print(action)
        print(state_node.state.belief.pos)
        print(state_node.state.belief.posteriors)
        print(state_node.state.locking)


if __name__ == '__main__':
    #cProfile.run("main()")
    main()
