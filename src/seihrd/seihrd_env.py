from typing import Sequence
import numpy as np
from gymnasium.envs.registration import EnvSpec
from seihrd.base_models import (
    State,
    Populations,
    SubCompartmentPopulations,
    Probs,
    SimHyperParams,
    A,
)
from random import random
import gymnasium as gym
from seihrd.stats import Stats
from seihrd.transitions.action_transitions import ActionTransitions
from seihrd.transitions.population_transitions import PopulationTransitions
from seihrd.transitions.seasonal_transitions import SeasonalTransitions


class SeihrdEnv(gym.Env):
    metadata = {'render.modes': ['ansi', 'human']}
    reward_range = (-np.inf, np.inf)
    spec = EnvSpec(
        id='seihrd-v0',
        entry_point='seihrd.seihrd_env:SeihrdEnv',
        max_episode_steps=365,
    )

    # Set these in ALL subclasses
    action_space = gym.spaces.MultiDiscrete([1, 1, 1, 1])
    observation_space = gym.spaces.Discrete(1)

    def __init__(self):
        self.state = self.get_initial_state()

        self.action_transitions = ActionTransitions()
        self.seasonal_transitions = SeasonalTransitions()
        self.population_transitions = PopulationTransitions()

        self.stats = Stats()

    def step(self, action: Sequence[int]):
        self.state = self.action_transitions(self.state, action)
        self.state = self.seasonal_transitions(self.state)
        self.state = self.population_transitions(self.state)
        # self.stats(self.state)

        self.state.step_count += 1
        self.state.is_done = self.state.step_count >= self.state.hyper_parameters.max_steps

        # Just for visualization, add noise to probs
        for prob_key in Probs.__fields__:
            noise = (random() - 0.5) * 0.05
            prob = getattr(self.state.probs, prob_key) + noise
            prob = max(0, prob)
            prob = min(1, prob)
            setattr(self.state.probs, prob_key, prob)

        # Just for visualization, add noise to epp
        self.state.epp += (random() - 0.5) * 0.01

    def reset(self):
        self.__init__()

    def get_state_dict(self):
        return self.state.dict()

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
                susceptible=SubCompartmentPopulations(uv=hp.initial_population),
                exposed1=SubCompartmentPopulations(),
                exposed2=SubCompartmentPopulations(),
                infected=SubCompartmentPopulations(),
                recovered=SubCompartmentPopulations(),
                hospitalized=SubCompartmentPopulations(),
                deceased=SubCompartmentPopulations(),
            ),
            probs=Probs(**{k: random() * 0.1 for k in Probs.__fields__}),
            action_residue={'vaccine': 1, 'mask': 0},
            epp=1.0,
            hyper_parameters=hp,
            action_mask={'noop': 1, 'vaccine': 0, 'mask': 1},
            step_count=0,
            is_done=False,
        )
        return state
