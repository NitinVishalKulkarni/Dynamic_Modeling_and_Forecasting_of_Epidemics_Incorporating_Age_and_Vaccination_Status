from typing import Sequence
import numpy as np
from seihrd.base_models import State, A, a2i


class ActionTransitions:
    """
    This will take the current state and the action and
    returns the updated state based on the action.
    Specifically, it updates the following properties:
        1. params.beta
        2. epp
        3. action_in_effect
        4. action_cool_down
        5. action_mask
    """

    def __init__(self):
        self.multiplier = {
            # S L  M  V    beta  epp
            (1, 0, 0, 0): (0.95, 0.9965),
            (0, 1, 0, 0): (0.85, 0.997),
            (0, 0, 1, 0): (0.925, 0.9965),
            (0, 0, 0, 1): (0.95, 0.994),
            (1, 1, 0, 0): (0.85, 0.997),  # L takes precedence.
            (1, 0, 1, 0): (0.875, 0.9965),
            (1, 0, 0, 1): (0.825, 0.993),
            (0, 1, 1, 0): (0.75, 0.994),
            (0, 1, 0, 1): (0.80, 0.993),
            (0, 1, 1, 1): (0.60, 0.9925),
            (1, 1, 1, 0): (0.75, 0.994),  # L takes precedence.
            (1, 1, 0, 1): (0.80, 0.993),  # L takes precedence.
            (1, 1, 1, 1): (0.60, 0.9925),  # L takes precedence.
            (0, 0, 1, 1): (0.90, 0.9935),
            (1, 0, 1, 1): (0.60, 0.9925),
        }

    def __call__(self, state: State, action: Sequence[int]):
        s = state.copy(deep=True)
        in_effect = np.array(s.action_in_effect, dtype=np.float)
        cooldown = np.array(s.action_cool_down, dtype=np.float)
        max_in_effect = np.array(s.hyper_parameters.action_durations)
        max_cooldown = np.array(s.hyper_parameters.action_cool_downs)

        # Only look at valid actions.
        action = np.array(s.action_mask) * np.array(action)
        # Apply actions that are already in effect.
        action = (action | (in_effect > 0)) * 1

        """ beta & epp """
        if action.sum() == 0:
            # TODO: Apply the weird logic for noop.
            beta_m = 1
            epp_m = 1
        else:
            beta_m, epp_m = self.multiplier[tuple(action)]

        s.params.beta *= beta_m
        s.epp *= epp_m

        """ action_in_effect """
        # Increment in_effect
        in_effect += action
        # If max duration reached, set in_effect = 0
        finished = max_in_effect == in_effect
        in_effect *= ~finished

        s.action_in_effect = in_effect

        """ cooldown """
        # Increment cool-downs for already cooling actions.
        cooldown += cooldown > 0
        # cool-down = 1 for newly finished actions.
        cooldown += finished
        # If max duration reached, set cooldown = 0
        cooldown *= max_cooldown != cooldown

        s.action_cool_down = cooldown

        """ action_mask """
        # An action is legal only if in_effect = 0 and cooldown = 0
        mask = (cooldown + in_effect) == 0
        # Additionally, if L is in_effect, S is illegal.
        if in_effect[a2i[A.lockdown]] > 0:
            mask[a2i[A.social_distancing]] = False

        s.action_mask = mask * 1

        return s


# TODO: Write tests for sanity check.
