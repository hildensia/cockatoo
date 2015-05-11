from __future__ import division

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

import argparse
import multiprocessing
from functools import partial

import collections
import datetime
import random

import os

import scipy.ndimage

try:
    import cPickle as pickle
except ImportError:
    import pickle


ExperimentData = collections.namedtuple(
    "ExperimentData", "action ll_action pos model_posterior locking_state "
                      "p_cp qs vs afs"
)

Metadata = collections.namedtuple(
    "Metadata", "intrinsic_motivation goal_reward mcts_n seed"
)

class Simulator(object):
    def __init__(self, noise, num_of_joints, tau=.01):
        self.world = World([])

        #  master | slave | open | closed
        # --------+-------+------+--------
        #    0    |   1   | 160+ |  160-
        #    1    |   2   | 160+ |  160-
        #    2    |   3   | 160+ |  160-
        #    3    |   4   | 160+ |  160-

        limits = (0, 180)

        for i in range(num_of_joints):
            dampings = [15, 200, 15]

            m = random.randint(10, 170)
            if i > 0:
                locks = [lower, upper]

            lower = (0, m - 10)
            upper = (m + 10, 180)

            self.world.add_joint(Joint([lower[1], upper[0]], dampings,
                                       limits=limits, noise=noise))
            if i > 0:
                MultiLocker(self.world, locker=self.world.joints[i-1],
                            locked=self.world.joints[i], locks=locks)

            print("Joint {} opens at {} - {}".format(i, lower[1], upper[0]))
        # for i in range(2, 5):
        #     MultiLocker(self.world, locker=self.world.joints[i-1],
        #                 locked=self.world.joints[i], locks=[closed])

        controllers = [Controller(self.world, j)
                       for j, _ in enumerate(self.world.joints)]
        self.action_machine = ActionMachine(self.world, controllers, tau)


def argparsing():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--action_n', '-n', type=int,
                        help='Number of processes to be started.',
                        default=20)
    parser.add_argument('--intrinsic_motivation', '-i', action='store_true',
                        help='Use intrinsic motivation')
    parser.add_argument('--goal_reward', '-g', action='store_true',
                        help='Give reward on the goal state.')
    parser.add_argument('--processes', '-p', type=int,
                        help='Number of processes to be started.',
                        default=None)
    parser.add_argument('--mcts_n', '-m', type=int,
                        help='Number of nodes MCTS should expand',
                        default=500)
    parser.add_argument('--seed', '-s', type=int,
                        help='Random number generator seed',
                        default=1)
    parser.add_argument('--output_directory', '-o', type=str,
                        help='Where to save the data files.',
                        default=os.getcwd())
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print("# Actions: {}".format(args.action_n))
    print("# MCTS nodes: {}".format(args.mcts_n))
    print("Intrinsic Motivation: {}".format("On" if args.intrinsic_motivation
                                            else "Off"))
    print("Goal Reward: {}".format("On" if args.goal_reward
                                   else "Off"))
    return args


def get_probability_over_degree(P, qs, prior):
    probs = np.zeros((360,))
    count = np.zeros((360,))
    deg = 0
    for i, pos in enumerate(qs):
        if i > 0:
            old_deg = deg

        deg = int(pos)%360

        probs[deg] += P[i]
        count[deg] += 1

        if i > 0 and abs(old_deg-deg) >= 2:
            bigger = max(deg, old_deg)
            smaller = min(deg, old_deg)
            if deg > old_deg:
                j = i-1
                k = i
            else:
                j = i
                k = i-1
            probs[smaller+1:bigger] = np.interp(range(smaller+1, bigger),
                                                [smaller, bigger],
                                                [P[j], P[k]])
            count[smaller+1:bigger] += 1

    probs = probs/count
    prior = .01
    probs = np.array([prior if np.isnan(p) else p for p in probs])
    return probs, count


def compute_p_same(p_cp):
    p_same = []
    for pcp in p_cp:
        p_same.append(same_segment(pcp))
    return np.array(p_same)


def get_damping_over_degree(P, qs):
    P = np.asarray(P)
    qs = np.asarray(qs)
    ds = np.zeros((360,))
    count = np.zeros((360,))
    for i, pos in enumerate(qs):
        deg = int(pos)%360

        ds[deg] += P[i]
        count[deg] += 1

    ds = ds/count
    old_d = 0
    for i, d in enumerate(ds):
        if np.isnan(d):
            ds[i] = old_d
        else:
            old_d = d

    return ds, count


def compute_damping(v, af, tau):
    v = np.asarray(v)
    af = np.asarray(af)
    vn = v + af
    d = np.zeros(v.shape)
    d[1:] = abs(((vn[:-1]**2) - (v[1:]**2))/(tau * vn[:-1]))

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

    return d


# def process_update_p_cp(tup):
#     (use_ros, records, j, prior) = tup
#     if use_ros:
#         q = records["q_" + str(j)].as_matrix()
#         af = records["applied_force_" + str(j)][0:].as_matrix()
#         v = q[1:] - q[:-1]  # we can't measure the velocity directly
#
#         vn = v[:] + af[1:]
#         d = np.zeros((v.shape[0] + 1,))
#         d[1:] = abs((vn**2 - v[:]**2)/(0.1 * vn))
#     else:
#         v = records["v_" + str(j)].as_matrix()
#         af = records["applied_force_" + str(j)].as_matrix()
#         vn = v + af
#         d = np.zeros(v.shape)
#         d[1:] = abs((vn[:-1]**2 - v[1:]**2)/(0.1 * vn[:-1]))
#
#         saved = 0
#         for i, vel in enumerate(v):
#             if vel == 0:
#                 d[i] = saved
#             else:
#                 saved = d[i]
#
#         d[0] = d[1]
#
#     nans, x = nan_helper(d)
#     if not all(nans):
#         d[nans] = np.interp(x(nans), x(~nans), d[~nans])
#
#     Q, P, Pcp = bcd.offline_changepoint_detection(
#         data=d,
#         prior_func=partial(bcd.const_prior, l=(len(d)+1)),
#         observation_log_likelihood_function=
#         bcd.gaussian_obs_log_likelihood,
#         truncate=-50)
#
#     p_cp, count = get_probability_over_degree(
#         np.exp(Pcp).sum(0)[:],
#         records['q_' + str(j)][1:].as_matrix(),
#         prior
#     )
#
#     return p_cp


def process_cp(d):
    Q, P, Pcp = bcd.offline_changepoint_detection(
        data=d,
        prior_func=partial(bcd.const_prior, l=(len(d)+1)),
        observation_log_likelihood_function=
        bcd.gaussian_obs_log_likelihood,
        truncate=-len(d)/2.3)

    return np.exp(Pcp).sum(0)


def process_update_p_cp(tup):
    use_ros, records, j = tup
    q = records["q_" + str(j)].as_matrix()
    if use_ros:
        af = records["applied_force_" + str(j)].as_matrix()[1:]
        v = q[1:] - q[:-1]  # we can't measure the velocity directly
    else:
        v = records["v_" + str(j)].as_matrix()
        af = records["applied_force_" + str(j)].as_matrix()

    damping = compute_damping(v, af, .01)
    damping_deg, counts = get_damping_over_degree(damping, q)

    pcpd = process_cp(damping_deg)
    pcpd[counts[1:] == 0] = np.maximum(.01, pcpd[counts[1:] == 0])

    pcpd = scipy.ndimage.filters.gaussian_filter1d(pcpd, 1)*2

    return pcpd


def update_p_cp(world, use_ros, pool):
    pid = multiprocessing.current_process().pid
    data = zip([use_ros] * len(world.joints),
               [Record.records[pid]] * len(world.joints),
               range(len(world.joints)))

    return pool.map(process_update_p_cp, data), Record.records[pid]


def init_model_prior(n):
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
    return model_prior


def init_experiences(n):
    experiences = [[]]*n
    print(experiences)
    for i in range(n):
        experiences[i] = [{"data": np.array([0]*n), "value": 1}]
    experiences[0][0]["value"] = 0
    return experiences


def main():
    # nicer numpy output
    np.set_printoptions(precision=3, suppress=True)

    # get command line arguments
    options = argparsing()

    # save metadata
    meta = Metadata(intrinsic_motivation=options.intrinsic_motivation,
                    goal_reward=options.goal_reward,
                    mcts_n=options.mcts_n,
                    seed=options.seed)
    number_of_joints = 5

    # init all experiences with the initial locking state
    experiences = init_experiences(number_of_joints)

    # setup priors
    # Changepoint prior
    p_cp = np.array([.01] * 360)
    #p_cp[160] = .9
    JointDependencyBelief.p_same = np.array([same_segment(p_cp)] *
                                            number_of_joints)
    JointDependencyBelief.alpha_prior = np.array([.1, .1])
    JointDependencyBelief.model_prior = init_model_prior(number_of_joints)

    #setup belief of first state
    belief = JointDependencyBelief(pos=[0] * number_of_joints,
                                   experiences=experiences)
    print("Pos: {}".format(belief.pos))

    # setup first state and initialize a simulator
    noise = {'q': 10e-6, 'vel': 10e-6}
    state = JointDependencyState(belief, [False, True, True, True, True],
                                 Simulator(noise, number_of_joints), options)

    print(np.asarray(belief.posteriors))

    # setup MCTS
    state_node = mcts.graph.StateNode(None, state)
    search = mcts.mcts.MCTS(tree_policy=mcts.tree_policies.UCB1(c=10),
                            default_policy=mcts.default_policies.
                            immediate_reward,
                            backup=mcts.backups.Bellman(gamma=.6))

    # setup container for experiment data
    data = []
    filename = os.path.join(
        options.output_directory,
        'cockatoo_{}_{}.pkl'.format(
            datetime.datetime.now().strftime("%y%m%d%H%M%S"),
            options.seed))

    print("Save to {}".format(filename))

    for _ in range(options.action_n):
        # reset multiprocessing pool
        pool = multiprocessing.Pool(processes=options.processes)
        JointDependencyBelief.pool = pool

        # search for best action
        action = search(state_node, n=options.mcts_n)
        print("Action: {}".format(action))
        ll_action = state_node.state.get_best_low_level_action(action)
        print("LL Action: {}".format(ll_action))

        # perform best action
        state_node = state_node.children[action].sample_state(real_world=True)

        # update change point probabilities from force sensor data
        p_cp, records = update_p_cp(state_node.state.simulator.world, False,
                                    pool=pool)
        JointDependencyBelief.p_same = compute_p_same(p_cp)

        # for i, pcp in enumerate(p_cp):
        #     plt.plot(pcp, label="{}".format(i))
        # plt.legend()
        # plt.grid()
        # plt.xticks(np.linspace(0,100,51))
        # plt.show()

        # set best state as new root node
        state_node.parent = None

        # print some interesting data
        print("Pos: {}".format(state_node.state.belief.pos))
        print("Model distribution: {}".format(
            np.asarray(state_node.state.belief.posteriors)))
        print("Locking state: {}".format(state_node.state.locking))

        qs = [records['q_{}'.format(q)] for q in range(number_of_joints)]
        vs = [records['v_{}'.format(q)] for q in range(number_of_joints)]
        afs = [records['applied_force_{}'.format(q)] for q in range(number_of_joints)]

        # pack data for later saving
        expd = ExperimentData(
            action=action,
            ll_action=ll_action,
            pos=state_node.state.belief.pos,
            model_posterior=state_node.state.belief.posteriors,
            locking_state=state_node.state.locking,
            p_cp=np.array(p_cp),
            qs=qs,
            vs=vs,
            afs=afs)
        data.append(expd)

        # shutdown workers from pool
        JointDependencyBelief.pool.close()
        JointDependencyBelief.pool.join()

        # save data
        with open(filename, 'wb') as f:
            pickle.dump((meta, data), f)

if __name__ == '__main__':
    main()
