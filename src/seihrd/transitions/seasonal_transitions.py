
class SeasonalTransitions:
    """
    This will take the current params and the current timestep and returns the updated params based on the timestep.
    """

    def __init__(self):
        # Multiplies e1_i and e2_i with multiplier at these timesteps.
        self.multiplier = {
            # time_step: multiplier,
            60: 1.1,
            90: 0.8,
            150: 1.2,
            225: 1.9,
            300: 0.4,
        }

    def __call__(self, state):
        # TODO: Implement seasonal transitions.
        if state.time_step in self.multiplier:
            state = state.copy(deep=True)

            multiplier = self.multiplier[state.time_step]

            state.params.e1_i *= multiplier
            state.params.e2_i *= multiplier

        return state
