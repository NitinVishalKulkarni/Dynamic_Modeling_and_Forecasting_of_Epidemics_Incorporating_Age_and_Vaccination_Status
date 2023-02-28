from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from src.epidemic_simulation_environment import EpidemicSimulation
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO


def create_env():
    env = EpidemicSimulation(
        data_path=f"../../../Data/Updated Data/epidemiological_model_data/",
        state_name="new_york",
        state_population=19_453_734,
        start_date="11/01/2021",
    )

    class SB3Observation(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
            )

        def observation(self, obs):
            return np.array(obs)

    env = SB3Observation(env)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    check_env(create_env())

    env = SubprocVecEnv([create_env] * 8, start_method="spawn")
    env = VecFrameStack(env, n_stack=14)
    print(env.reset().shape)

    # model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    # print(model.policy)

    start = time.time()
    model.learn(total_timesteps=100_000)
    print("Runtime:", time.time() - start)
