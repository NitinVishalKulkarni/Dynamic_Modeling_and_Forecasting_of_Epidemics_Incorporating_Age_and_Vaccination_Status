import sys
import ray
from ray.rllib.algorithms import ppo, sac
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy, PPOTF2Policy
from ray.rllib.algorithms.sac import SACConfig, SACTorchPolicy, SACTFPolicy, RNNSACTorchPolicy, RNNSACConfig
from src.epidemic_simulation_environment.epidemic_simulation_environment import EpidemicSimulation
import time

environment_configuration = {'data_path': f"../../../Data/Updated Data/epidemiological_model_data/",
                             'state_name': 'new_york', 'state_population': 19_453_734, 'start_date': '11/01/2021'}

# # Recurrent SAC
# algorithm_config = RNNSACConfig().training(gamma=0.9, lr=0.01)
# algorithm_config = algorithm_config.resources(num_gpus=0)
# algorithm_config = algorithm_config.rollouts(num_rollout_workers=4)
# # algorithm_config.model['use_lstm'] = True
# print('Algorithm Configuration:\n', algorithm_config.to_dict())
# print('\nQ Network:\n', algorithm_config.q_model_config)
# print('\nPolicy Model:', algorithm_config.policy_model_config)
#
# # Build an Algorithm object from the config and run 1 training iteration.
# algorithm = algorithm_config.build(env="CartPole-v1")
# policy = algorithm.get_policy()
# print('\nModel Details:')
# print('Framework:', policy.framework)
# print('Is policy recurrent:', policy.is_recurrent())
# print('Observation Space:', policy.observation_space)
# print('Observation Space Structure:', policy.observation_space_struct)
# print('Action Space:', policy.action_space)
# print('Action Space Structure:', policy.action_space_struct)
# print('SAC Model Summary:', policy)
# # algorithm.train()
# sys.exit()

# Sequential SAC
algorithm_config = SACConfig()
# algorithm_config = algorithm_config.training(gamma=0.99, lr=0.01, kl_coeff=0.3)
algorithm_config = algorithm_config.training(gamma=0.99, lr=0.01)
algorithm_config = algorithm_config.resources(num_gpus=0)
algorithm_config = algorithm_config.rollouts(num_rollout_workers=1)
algorithm_config = algorithm_config.evaluation(evaluation_num_workers=1)
algorithm_config = algorithm_config.environment(EpidemicSimulation, env_config=environment_configuration)

print('Algorithm Configuration:\n', algorithm_config.to_dict())
print('\nSAC Q Network:\n', algorithm_config.q_model_config)
print('\nSAC Policy Model:', algorithm_config.policy_model_config)

# ray.rllib.utils.check_env(EpidemicSimulation(env_config=environment_configuration))
# sys.exit()

# Build an algorithm object from the config and run 1 training iteration.
algorithm = algorithm_config.build()
policy = algorithm.get_policy()
print('\nModel Details:')
print('Framework:', policy.framework)
print('Is policy recurrent:', policy.is_recurrent())
print('Observation Space:', policy.observation_space)
print('Observation Space Structure:', policy.observation_space_struct)
print('Action Space:', policy.action_space)
print('Action Space Structure:', policy.action_space_struct)
# print('PPO Model Summary:', policy.model.base_model.summary())
print('SAC Model Summary:', policy)
# sys.exit()

training_start_time = time.time()
for training_iteration in range(10):
    print(f'\nIteration {training_iteration + 1}:\n', algorithm.train())
training_end_time = time.time()

print('\nTraining Time:', training_end_time - training_start_time)

print('\nEvaluation:\n', algorithm.evaluate())
