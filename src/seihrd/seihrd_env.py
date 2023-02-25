from dataclasses import asdict
from typing import Sequence
import numpy as np
from gymnasium.envs.registration import EnvSpec
from seihrd.base_models import (
    State,
    Populations,
    SubCompPopulations,
    Params,
    SimHyperParams,
    A, SubCompParams,
)
from random import random
import gymnasium as gym
from seihrd.transitions.action_transitions import ActionTransitions
from seihrd.transitions.population_transitions import PopulationTransitions
from seihrd.transitions.seasonal_transitions import SeasonalTransitions


class SeihrdEnv(gym.Env):
    metadata = {'render.modes': ['ansi', 'human']}
    reward_range = (-np.inf, np.inf)
    action_space = gym.spaces.MultiDiscrete([1, 1, 1, 1])
    observation_space = gym.spaces.Discrete(1)

    spec = EnvSpec(
        id='seihrd-v0',
        entry_point='seihrd.seihrd_env:SeihrdEnv',
        max_episode_steps=365,
    )

    def __init__(self):
        self.state = self.get_initial_state()

        self.action_transitions = ActionTransitions()
        self.seasonal_transitions = SeasonalTransitions()
        self.population_transitions = PopulationTransitions()

        self.action_mask = np.zeros(4)

    def step(self, action: Sequence[int]):
        self.state = self.action_transitions(self.state, action)
        self.state = self.seasonal_transitions(self.state)
        self.state = self.population_transitions(self.state)

        self.state.step_count += 1
        self.state.is_done = self.state.step_count >= self.state.hyper_parameters.max_steps

        # TODO: Update action_mask
        # TODO: Reward computation
        # TODO: Observation from state

        # Just for visualization, add noise to params
        from dataclasses import fields
        for prob_key in fields(Params):
            noise = (random() - 0.5) * 0.05
            prob = getattr(self.state.params, prob_key.name) + noise
            prob = max(0, prob)
            prob = min(1, prob)
            setattr(self.state.params, prob_key.name, prob)

        # Just for visualization, add noise to epp
        self.state.epp += (random() - 0.5) * 0.01

    def reset(self):
        self.__init__()

    def get_state_dict(self):
        return asdict(self.state)

    @staticmethod
    def get_initial_state():
        hp = SimHyperParams(
            action_durations={A.vaccine: 14, A.mask: 150, A.lockdown: 14, A.social_distancing: 30},
            action_cool_downs={A.vaccine: 14, A.mask: 150, A.lockdown: 14, A.social_distancing: 30},
            max_steps=365,
            initial_population=1000,
        )
        state = State(
            populations=Populations(
                susceptible=SubCompPopulations(uv=hp.initial_population),
                exposed=SubCompPopulations(),
                infected=SubCompPopulations(),
                recovered=SubCompPopulations(),
                hospitalized=SubCompPopulations(),
                deceased=SubCompPopulations(),
            ),
            params=Params(
                vfv=random(),
                vb=random(),
                alpha=random(),
                beta=random(),
                e_s=SubCompParams(uv=random(), fv=random(), b=random()),
                e_i=SubCompParams(uv=random(), fv=random(), b=random()),
                i_r=SubCompParams(uv=random(), fv=random(), b=random()),
                i_h=SubCompParams(uv=random(), fv=random(), b=random()),
                i_d=SubCompParams(uv=random(), fv=random(), b=random()),
                e2_i=SubCompParams(uv=random(), fv=random(), b=random()),
                h_r=SubCompParams(uv=random(), fv=random(), b=random()),
                h_d=SubCompParams(uv=random(), fv=random(), b=random()),
                e_r=SubCompParams(uv=random(), fv=random(), b=random()),
            ),
            action_residue={'vaccine': 1, 'mask': 0},
            epp=1.0,
            hyper_parameters=hp,
            action_mask={'noop': 1, 'vaccine': 0, 'mask': 1},
            step_count=0,
            is_done=False,
        )
        return state


if __name__ == '__main__':
    print(SeihrdEnv().state.populations['susceptible'])
