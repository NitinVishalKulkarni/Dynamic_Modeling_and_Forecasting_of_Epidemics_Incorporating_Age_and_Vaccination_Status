import unittest
from seihrd.sim.seihrd_env import SeihrdEnv


class ActionMaskTestCase(unittest.TestCase):
    def test_initial_mask(self):
        env = SeihrdEnv()
        env.reset()
        self.assertEqual(list(env.state.action_mask), [1, 1, 1, 1])

        # Taking no action should keep the action mask the same.
        env.step([0, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [1, 1, 1, 1])

        # No matter how many times you take no action, masks will be 1,1,1,1
        for _ in range(10):
            env.step([0, 0, 0, 0])
            self.assertEqual(list(env.state.action_mask), [1, 1, 1, 1])

    def test_illegal_action(self):
        env = SeihrdEnv()
        env.reset()
        env.state.hyper_parameters.action_durations = [4, 4, 4, 4]
        env.state.hyper_parameters.action_cool_downs = [4, 4, 4, 4]
        env.step([0, 0, 0, 0])

        """ TAKING THE ACTION """
        env.step([1, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [0, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [1, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [0, 0, 0, 0])

        """ IN EFFECT """
        env.step([0, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [0, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [2, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [0, 0, 0, 0])

        env.step([0, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [0, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [3, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [0, 0, 0, 0])

        """ COOL DOWN """
        env.step([0, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [0, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [0, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [1, 0, 0, 0])

        env.step([0, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [0, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [0, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [2, 0, 0, 0])

        env.step([0, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [0, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [0, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [3, 0, 0, 0])

        """ ACTION ELIGIBLE TO TAKE AGAIN """
        env.step([0, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [1, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [0, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [0, 0, 0, 0])

        env.step([0, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [1, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [0, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [0, 0, 0, 0])

        """ TAKE ACTION AGAIN """
        env.step([1, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [0, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [1, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [0, 0, 0, 0])

        """ TAKE ILLEGAL ACTION """
        env.step([1, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [0, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [2, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [0, 0, 0, 0])

    def test_lockdown_disables_social_distancing(self):
        env = SeihrdEnv()
        env.reset()

        env.step([0, 0, 0, 0])
        self.assertEqual(list(env.state.action_mask), [1, 1, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [0, 0, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [0, 0, 0, 0])

        env.step([0, 1, 0, 0])
        self.assertEqual(list(env.state.action_mask), [0, 0, 1, 1])
        self.assertEqual(list(env.state.action_in_effect), [0, 1, 0, 0])
        self.assertEqual(list(env.state.action_cool_down), [0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
