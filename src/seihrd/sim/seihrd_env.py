from typing import Sequence
import numpy as np
from gymnasium.envs.registration import EnvSpec
from seihrd.sim.base_models import (
    State,
    Populations,
    SubCompPopulations,
    Params,
    SimHyperParams,
    SubCompParams,
    i2a,
)
from random import random
import gymnasium as gym
from seihrd.sim.transitions.action_transitions import ActionTransitions
from seihrd.sim.transitions.population_transitions import PopulationTransitions
from seihrd.sim.transitions.seasonal_transitions import SeasonalTransitions


class SeihrdEnv(gym.Env):
    metadata = {'render.modes': ['ansi', 'human']}
    reward_range = (-np.inf, np.inf)
    action_space = gym.spaces.MultiDiscrete([2, 2, 2, 2])
    observation_space = gym.spaces.Discrete(24)

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

    def step(self, action: Sequence[int]):
        s = self.state
        prev_infected = s.populations.infected.total()

        s = self.seasonal_transitions(s)
        s = self.action_transitions(s, action)
        s = self.population_transitions(s)

        s.time_step += 1
        s.is_done = s.time_step >= s.hyper_parameters.max_steps

        # Reward calculation
        if prev_infected == 0:
            infection_change = 0
        else:
            infection_change = (prev_infected - s.populations.infected.total()) / prev_infected
        reward = infection_change + s.epp

        self.state = s
        return self.observe(), reward, s.is_done, s.is_done, {'action_mask': s.action_mask}

    def reset(self, *_args, **_kwargs) -> (np.array, dict):
        self.__init__()
        return self.observe(), {'action_mask': self.state.action_mask}

    def observe(self):
        populations = self.state.populations.to_list()
        populations = np.array(populations).flatten() / self.state.populations.total()

        in_affect = [self.state.action_in_effect[a] / self.state.hyper_parameters.action_durations[a] for a in range(len(i2a))]
        progress = self.state.time_step / self.state.hyper_parameters.max_steps

        obs = np.concatenate((populations, in_affect, [progress, self.state.epp]))
        return obs

    def render(self, mode='human'):
        s = self.state.json(indent=4)
        if mode == 'human':
            print(s)
        elif mode == 'ansi':
            return s
        else:
            print(f'Render models can only be one of: {self.metadata["render.modes"]}')
            self.render('human')

    def get_state_dict(self):
        return self.state.dict()

    def get_masks(self):
        mask = np.ones((len(env.state.action_mask), 2))
        mask[:, 1] = self.state.action_mask
        mask = mask.astype(np.int8)
        mask = tuple(mask)
        return mask

    @staticmethod
    def get_initial_state():
        # TODO: AD: Get the initial state.
        hp = SimHyperParams(
            action_durations=[14, 150, 14, 30],
            action_cool_downs=[14, 150, 14, 30],
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
            epp=1.0,
            hyper_parameters=hp,
            action_in_effect=[0, 0, 0, 0],
            action_cool_down=[0, 0, 0, 0],
            action_mask=[1, 1, 1, 1],
            time_step=0,
            is_done=False,
        )
        return state


if __name__ == '__main__':
    env = SeihrdEnv()
    env.reset()
    for _ in range(10):
        action = env.action_space.sample(mask=env.get_masks())
        env.step(action)
        print(action, env.state.action_mask)
