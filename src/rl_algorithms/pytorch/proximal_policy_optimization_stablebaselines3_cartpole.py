import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time

# Parallel environments
number_of_training_environments = 1
training_environment = make_vec_env(
    "CartPole-v1",
    n_envs=number_of_training_environments,
    env_kwargs={"max_episode_steps": 10000},
)

model = PPO("MlpPolicy", training_environment, verbose=1)

start = time.time()
model.learn(total_timesteps=100 * 10000)
print("Training Time:", time.time() - start)

# model.save("ppo_cartpole")
# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")

# test_environment = gym.make("CartPole-v1", max_episode_steps=10000)
total_rewards = []

# for _ in range(100):
test_observation = training_environment.reset()
# zero_completed_observation = np.zeros(
#     (number_of_training_environments,)
#     + training_environment.observation_space.shape
# )
# zero_completed_observation[0, :] = test_observation
dones = [False]
total_reward = 0
while not np.any(dones):
    action, _states = model.predict(test_observation)

    test_observation, rewards, dones, info = training_environment.step(action)
    total_reward += 1
total_rewards.append(total_reward)

print("Total Reward:", np.mean(total_rewards), np.median(total_rewards))
