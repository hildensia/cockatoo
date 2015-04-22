from cockatoo.mcts_states import JointDependencyBelief, JointDependencyState
from cockatoo.utils import nan_helper

from joint_dependency.inference import same_segment
from joint_dependency.simulation import (Joint, MultiLocker, Controller,
                                         ActionMachine, World)
from joint_dependency.recorder import Record

import bayesian_changepoint_detection.offline_changepoint_detection as bcd
import numpy as np
import mcts.graph
import mcts.mcts
import mcts.backups
import mcts.tree_policies
import mcts.default_policies

import cProfile
import argparse
import multiprocessing
from functools import partial


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


def argparsing():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--intrinsic_motivation', '-i', action='store_true',
                        help='Use intrinsic motivation')
    parser.add_argument('--goal_reward', '-g', action='store_true',
                        help='Give reward on the goal state.')
    args = parser.parse_args()
    print("Intrinsic Motivation: {}".format("On" if args.intrinsic_motivation
                                            else "Off"))
    print("Goal Reward: {}".format("On" if args.goal_reward
                                   else "Off"))
    return args


def get_probability_over_degree(P, qs):
    probs = np.zeros((360,))
    count = np.zeros((360,))
    for i, pos in enumerate(qs[:-2]):

        deg = int(pos)%360

        probs[deg] += P[i]
        count[deg] += 1

    probs = probs/count
    prior = 10e-8
    probs = np.array([prior if np.isnan(p) else p for p in probs])
    return probs, count


def compute_p_same(p_cp):
    p_same = []
    for pcp in p_cp:
        p_same.append(same_segment(pcp))
    return p_same


def process_update_p_cp(tup):
    (use_ros, records, j) = tup
    if use_ros:
        q = records["q_" + str(j)].as_matrix()
        af = records["applied_force_" + str(j)][0:].as_matrix()
        v = q[1:] - q[:-1]  # we can't measure the velocity directly

        vn = v[:] + af[1:]
        d = np.zeros((v.shape[0] + 1,))
        d[1:] = abs((vn**2 - v[:]**2)/(0.1 * vn))
    else:
        v = records["v_" + str(j)].as_matrix()
        af = records["applied_force_" + str(j)].as_matrix()
        vn = v + af
        d = np.zeros(v.shape)
        d[1:] = (vn[:-1]**2 - v[1:]**2)/(0.1 * vn[:-1])

        saved = 0
        for i, vel in enumerate(v):
            if vel == 0:
                d[i] = saved
            else:
                saved = d[i]

        d[0] = d[1]

    nans, x = nan_helper(d)
    if not all(nans):
        d[nans] = np.interp(x(nans), x(~nans), d[~nans])

    Q, P, Pcp = bcd.offline_changepoint_detection(
        data=d,
        prior_func=partial(bcd.const_prior, l=(len(d)+1)),
        observation_log_likelihood_function=
        bcd.gaussian_obs_log_likelihood,
        truncate=-50)

    p_cp, count = get_probability_over_degree(
        np.exp(Pcp).sum(0)[:],
        records['q_' + str(j)][1:].as_matrix())

    return p_cp

def update_p_cp(world, use_ros, pool):
    pid = multiprocessing.current_process().pid
    print("PID: {}".format(pid))
    print(Record.records.keys())

    data = zip([use_ros] * len(world.joints),
               [Record.records[pid]] * len(world.joints),
               range(len(world.joints)))

    return pool.map(process_update_p_cp, data)


def main():
    options = argparsing()
    n = 5
    experiences = [[]]*n
    print(experiences)
    for i in range(n):
        experiences[i] = [{"data": np.array([0]*n), "value": 1}]
    experiences[0][0]["value"] = 0

    print(experiences)

    pool = multiprocessing.Pool(processes=4)
    JointDependencyBelief.pool = pool

    belief = JointDependencyBelief([0] * n, experiences)
    print("Pos: {}".format(belief.pos))

    belief.alpha_prior = np.array([.1, .1])

    p_cp = np.array([.3] * 360)
    #p_cp[160] = .9

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
                                 Simulator(noise), options)

    print(belief.posteriors)

    root = mcts.graph.StateNode(None, state)
    search = mcts.mcts.MCTS(tree_policy=mcts.tree_policies.UCB1(c=10),
                            default_policy=mcts.default_policies.immediate_reward,
                            backup=mcts.backups.Bellman(gamma=.6))

    state_node = root
    for _ in range(1):
        action = search(state_node, n=500)
        state_node = state_node.children[action].sample_state(real_world=True)
        p_cp = update_p_cp(state_node.state.simulator.world, False, pool=pool)
        # plt.plot(p_cp[0])
        # plt.show()

        # TODO: copy to all states
        state_node.state.belief.p_same = compute_p_same(p_cp)

        state_node.parent = None
        print(p_cp)
        print(action)
        print(state_node.state.belief.pos)
        print(state_node.state.belief.posteriors)
        print(state_node.state.locking)


if __name__ == '__main__':
    cProfile.run("main()")
    # main()
