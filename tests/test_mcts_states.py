import pytest
from cockatoo.mcts_states import (JointDependencyState, JointDependencyAction,
                                  JointDependencyActionType)
from cockatoo.belief import JointDependencyBelief
from joint_dependency.inference import same_segment, prob_locked
import multiprocessing
import numpy as np

parametrize_n = pytest.mark.parametrize("n", [100, 1000])


@pytest.fixture
def state():
    experiences = [[], []]
    for i in range(2):
        experiences[i].append({"data": np.array([0, 0]),
                               "value": 1})
    experiences[-1][0]["value"] = 0

    belief = JointDependencyBelief([0, 0], experiences)
    JointDependencyBelief.model_prior = np.array([[0, .5, .5], [.5, 0, .5]])
    JointDependencyBelief.alpha_prior = np.array([.1, .1])
    p_cp = np.array([.0001] * 360)
    p_cp[10] = .99
    p_cp[160] = .99
    JointDependencyBelief.p_same = [same_segment(p_cp), same_segment(p_cp)]
    JointDependencyBelief.pool = multiprocessing.Pool(processes=1)
    return JointDependencyState(belief, [1, 0], None, None)


@pytest.mark.parametrize("action_type", JointDependencyActionType)
@parametrize_n
def test_perform(state, n, action_type):
    action = JointDependencyAction(action_type, 0)
    ones = 0
    for _ in range(n):
        new_state = state.perform(action)
        assert len(new_state.belief.experiences) == 2
        assert len(new_state.belief.experiences[0]) == 2
        assert len(new_state.belief.experiences[1]) == 2
        if new_state.belief.experiences[0][1]["value"] == 1:
            ones += 1

    best_action = state.get_best_low_level_action(action)
    pl = prob_locked(state.belief.experiences[0],
                     best_action,
                     state.belief.p_same,
                     state.belief.alpha_prior,
                     state.belief.model_prior[0]).mean()

    assert ((n * pl[0]) - (3 * n/np.sqrt(n)) < ones
            < (n * pl[0]) + (3 * n/np.sqrt(n)))

    print(new_state.belief.experiences)

@pytest.mark.parametrize("action_type", JointDependencyActionType)
@pytest.mark.parametrize("j", range(5))
def test_action_hashing(action_type, j):
    a1 = JointDependencyAction(action_type, j)
    a2 = JointDependencyAction(action_type, j)

    assert a1 == a2
    assert a1 == a1

    d = {a1: 12}
    assert d[a1] == 12
    assert d[a2] == 12




