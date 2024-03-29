from typing import Sequence
import numpy as np
from seihrd.sim.base_models import State, A, a2i


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
        6. params.vfv
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
        in_effect = np.array(s.action_in_effect, dtype=float)
        cooldown = np.array(s.action_cool_down, dtype=float)
        max_in_effect = np.array(s.hyper_parameters.action_durations)
        max_cooldown = np.array(s.hyper_parameters.action_cool_downs)

        # Only look at valid actions.
        action = np.array(s.action_mask) * np.array(action)
        # Apply actions that are already in effect.
        action = (action | (in_effect > 0)) * 1

        """ BETA & EPP """
        if action.sum() == 0:
            if s.populations.infected.total() / s.populations.total() < 0.001:
                beta_m = 1.1
                epp_m = 1.005
            else:
                beta_m = 1.4
                epp_m = 0.999
        else:
            beta_m, epp_m = self.multiplier[tuple(action)]

        s.params.beta *= beta_m
        s.epp *= epp_m
        s.epp = min(s.epp, 100)

        """ IN EFFECT """
        # Increment in_effect
        in_effect += action
        # If max duration reached, set in_effect = 0
        finished = max_in_effect == in_effect
        in_effect *= ~finished

        s.action_in_effect = in_effect
        # AD: Should we reverse the effect after the in_effect is done?

        """ COOLDOWN """
        # Increment cool-downs for already cooling actions.
        cooldown += cooldown > 0
        # cool-down = 1 for newly finished actions.
        cooldown += finished
        # If max duration reached, set cooldown = 0
        cooldown *= max_cooldown != cooldown

        s.action_cool_down = cooldown

        """ ACTION MASK """
        # An action is legal only if in_effect = 0 and cooldown = 0
        mask = (cooldown + in_effect) == 0
        # Additionally, if L is in_effect, S is illegal.
        if in_effect[a2i[A.lockdown]] > 0:
            mask[a2i[A.social_distancing]] = False

        s.action_mask = mask * 1

        """ VFV """
        # AD: Nitin, can you please explain?
        if tuple(action) in (  # M, SM, SV, LM
            # S L  M  V
            (0, 0, 1, 0),
            (1, 0, 1, 0),
            (1, 0, 0, 1),
            (0, 1, 1, 0),
        ):
            s.params.vfv = 0.007084760245099044

        return s
