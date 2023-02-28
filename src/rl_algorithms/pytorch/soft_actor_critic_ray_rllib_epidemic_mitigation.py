import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from ray.rllib.algorithms.sac import SACConfig, SACTorchPolicy
from src.epidemic_simulation_environment import EpidemicSimulation

env = EpidemicSimulation(
        data_path=f"../../../Data/Updated Data/epidemiological_model_data/",
        state_name="new_york",
        state_population=19_453_734,
        start_date="11/01/2021",
    )

config = PPOConfig()
config = config.training(gamma=0.99, lr=0.01, kl_coeff=0.3)
config = config.resources(num_gpus=1)
config = config.rollouts(num_rollout_workers=4)
config.evaluation(evaluation_num_workers=1)
print(config.to_dict())

# Build an algorithm object from the config and run 1 training iteration.
algo = config.build(env=env)
for _ in range(5):
    print(algo.train())  # 3. train it,

algo.evaluate()
