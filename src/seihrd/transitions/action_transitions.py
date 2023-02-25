from seihrd.base_models import A, i2a


class ActionTransitions:
    """
    This will take the current params and the action and returns the updated params based on the action.
    """

    def __init__(self):
        """
        Let's say this is a simple multiplier of e1_i and e2_i.
        Next, we can add vaccination multiplier for vaccination action.
        TODO: Add action's impact on epp
        TODO: Update the action residue
        TODO: set invalid actions to 0
        """
        self.infection_multiplier = {
            A.social_distancing: 1.01,
            A.lockdown: 1.05,
            A.mask: 1.01,
            A.vaccine: 1.05,
        }

    def __call__(self, state, action):
        state = state.copy(deep=True)

        multiplier = 1
        for i, a in enumerate(i2a):
            if action[i]:
                multiplier *= self.infection_multiplier[a]

        state.params.e1_i *= multiplier
        state.params.e2_i *= multiplier

        return state
