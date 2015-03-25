from joint_dependency.simulation import (Controller, ActionMachine)


class JointDependencySimulator(object):
    def __init__(self, world):
        self.world = world
        self.controller = [Controller(world, j)
                           for j, _ in enumerate(world.joints)]
        self.action_machine = ActionMachine(self.world, self.controller)

    def perform(self, action):
        self.action_machine.run_action(action)

    def locking_state(self):
        return [j.locked for j in self.world.joints]
