import multiprocessing
import numpy as np

from joint_dependency.simulation import Record

from .utils import nan_helper


class JointDependencyBelief(object):
    def __init__(self, sim_state, experiences, world):
        self.experiences = experiences

        self.p_same = [[], []]
        self.p_dependencies = [[], []]
        self.p_locking = [[], []]

        self.world = world

        self.alpha_prior = 0
        self.model_prior = 0

    def sample(self):
        # what to sample?
        #  - segment borders
        #  - dependency model
        #  - locking state
        pass




