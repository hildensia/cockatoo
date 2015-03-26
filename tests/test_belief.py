import pytest
import numpy as np
from cockatoo.belief import *
from joint_dependency.inference import (same_segment, likelihood_dependent,
                                        likelihood_independent)


parametrize_n = pytest.mark.parametrize("n", [100, 1000, 10000])

@pytest.fixture
def belief_2d():
    belief = JointDependencyBelief([12, 14], [[], []])
    belief.model_prior = np.array([[0, .5, .5], [.5, 0, .5]])
    belief.alpha_prior = np.array([.1, .1])
    p_cp = np.array([.0001] * 360)
    p_cp[10] = .99
    p_cp[160] = .99
    belief.p_same = [same_segment(p_cp), same_segment(p_cp)]

    return belief


def test_empty_posterior(belief_2d):
    np.random.seed()
    print belief_2d.posteriors
    assert len(belief_2d.posteriors) == 2
    assert (belief_2d.posteriors == belief_2d.model_prior).all()


def test_experiences(belief_2d):

    belief_2d.experiences[1].append({"data": np.array([12, 14]), "value": 0})
    belief_2d.experiences[0].append({"data": np.array([12, 14]), "value": 0})
    belief_2d.experiences[0].append({"data": np.array([12, 2]), "value": 1})
    belief_2d.experiences[0].append({"data": np.array([12, 50]), "value": 0})
    belief_2d.experiences[0].append({"data": np.array([12, 170]), "value": 1})

    print("Belief: {}".format(belief_2d.posteriors))

    assert belief_2d.posteriors[0][1] > .9
    assert (belief_2d.posteriors[1] == np.array([.5, 0., .5])).all()


@parametrize_n
def test_sample_lock(belief_2d, n):
    np.random.seed()
    result = {True: 0, False: 0}
    for _ in range(n):
        result[belief_2d.sample_locking([12, 53])[0]] += 1

    assert (0.5 * n - (3 * n / np.sqrt(n)) < result[True]
            < 0.5 * n + (3 * n / np.sqrt(n)))


@parametrize_n
def test_simulate(belief_2d, n):
    np.random.seed()
    result = {0: 0, 1: 0}
    for _ in range(n):
        experience = belief_2d.simulate(np.array([12, 20]), 0)
        result[experience["value"]] += 1

    assert (0.5 * n - (3 * n / np.sqrt(n)) < result[0]
            < 0.5 * n + (3 * n / np.sqrt(n)))


