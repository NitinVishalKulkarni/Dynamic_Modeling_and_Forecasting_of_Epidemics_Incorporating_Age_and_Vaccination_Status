import pathlib
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input, LSTM
from tensorflow.python.keras.optimizer_v2.adam import Adam

from src.epidemic_simulation_environment import EpidemicSimulation


# noinspection DuplicatedCode
class PPOBuffer:
    """This class implements the PPO Buffer for storing the trajectories."""

    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        """This method initializes the PPO buffer."""

        self.observation_buffer = np.zeros(
            (size, observation_dimensions[0], observation_dimensions[1]),
            dtype=np.float32,
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.log_probability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, log_probability):
        """This method appends the transition to the PPO buffer."""

        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.log_probability_buffer[self.pointer] = log_probability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        """This method finishes the trajectory by computing advantage estimates and rewards-to-go."""

        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        """This method normalizes the advantages and returns the PPO buffer."""

        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.log_probability_buffer,
        )

    @staticmethod
    def discounted_cumulative_sums(x, discount):
        """This method computes the discounted cumulative sums of vectors for computing rewards-to-go
        and advantage estimates"""

        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# noinspection DuplicatedCode
class ProximalPolicyOptimization:
    """This class implements the Proximal Policy Optimization algorithm in Tensorflow."""

    def __init__(self, environment, initialization_parameters):
        """This method initializes the AWR parameters, and calls the train, evaluate and render_actions methods.

        :param environment: Gymnasium Environment - Environment the agent will be trained on.
        :param initialization_parameters: Dictionary - Dictionary containing the initialization parameters for the PPO
                                                       algorithm."""

        # Saving the training results.
        self.date_and_time = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        pathlib.Path(f"./Results/DataFrames/{self.date_and_time}").mkdir(
            parents=True, exist_ok=True
        )

        # Hyperparameters of the PPO algorithm.
        self.steps_per_epoch = initialization_parameters["steps_per_epoch"]
        self.number_of_epochs = initialization_parameters["number_of_epochs"]
        self.gamma = initialization_parameters["gamma"]
        self.clip_ratio = initialization_parameters["clip_ratio"]
        self.policy_learning_rate = initialization_parameters["policy_learning_rate"]
        self.value_function_learning_rate = initialization_parameters[
            "value_function_learning_rate"
        ]
        self.train_policy_iterations = initialization_parameters[
            "train_policy_iteration"
        ]
        self.train_value_iterations = initialization_parameters[
            "train_value_iterations"
        ]
        self.lam = initialization_parameters["lam"]
        self.target_kl = initialization_parameters["target_kl"]

        self.environment = environment
        self.time_period = initialization_parameters["time_period"]
        self.observation_dimensions = (
            self.time_period,
            self.environment.observation_space.n,
        )
        self.number_of_actions = self.environment.action_space.n

        self.buffer = PPOBuffer(self.observation_dimensions, self.steps_per_epoch)

        (
            self.actor,
            self.critic,
            self.policy_optimizer,
            self.value_optimizer,
        ) = self.build_neural_network()

        # Lists for plotting:
        # Susceptible:
        self.number_of_unvaccinated_susceptible_individuals_list = [
            self.environment.number_of_unvaccinated_susceptible_individuals
        ]
        self.number_of_fully_vaccinated_susceptible_individuals_list = [
            self.environment.number_of_fully_vaccinated_susceptible_individuals
        ]
        self.number_of_booster_vaccinated_susceptible_individuals_list = [
            self.environment.number_of_booster_vaccinated_susceptible_individuals
        ]
        self.number_of_susceptible_individuals_list = [
            self.environment.number_of_susceptible_individuals
        ]

        # Exposed:
        self.number_of_unvaccinated_exposed_individuals_list = [
            self.environment.number_of_unvaccinated_exposed_individuals
        ]
        self.number_of_fully_vaccinated_exposed_individuals_list = [
            self.environment.number_of_fully_vaccinated_exposed_individuals
        ]
        self.number_of_booster_vaccinated_exposed_individuals_list = [
            self.environment.number_of_booster_vaccinated_exposed_individuals
        ]
        self.number_of_exposed_individuals_list = [
            self.environment.number_of_exposed_individuals
        ]

        # Infected:
        self.number_of_unvaccinated_infected_individuals_list = [
            self.environment.number_of_unvaccinated_infected_individuals
        ]
        self.number_of_fully_vaccinated_infected_individuals_list = [
            self.environment.number_of_fully_vaccinated_infected_individuals
        ]
        self.number_of_booster_vaccinated_infected_individuals_list = [
            self.environment.number_of_booster_vaccinated_infected_individuals
        ]
        self.number_of_infected_individuals_list = [
            self.environment.number_of_infected_individuals
        ]

        # Hospitalized:
        self.number_of_unvaccinated_hospitalized_individuals_list = [
            self.environment.number_of_unvaccinated_hospitalized_individuals
        ]
        self.number_of_fully_vaccinated_hospitalized_individuals_list = [
            self.environment.number_of_fully_vaccinated_hospitalized_individuals
        ]
        self.number_of_booster_vaccinated_hospitalized_individuals_list = [
            self.environment.number_of_booster_vaccinated_hospitalized_individuals
        ]
        self.number_of_hospitalized_individuals_list = [
            self.environment.number_of_hospitalized_individuals
        ]

        # Recovered:
        self.number_of_unvaccinated_recovered_individuals_list = [
            self.environment.number_of_unvaccinated_recovered_individuals
        ]
        self.number_of_fully_vaccinated_recovered_individuals_list = [
            self.environment.number_of_fully_vaccinated_recovered_individuals
        ]
        self.number_of_booster_vaccinated_recovered_individuals_list = [
            self.environment.number_of_booster_vaccinated_recovered_individuals
        ]
        self.number_of_recovered_individuals_list = [
            self.environment.number_of_recovered_individuals
        ]

        # Deceased:
        self.number_of_unvaccinated_deceased_individuals_list = [
            self.environment.number_of_unvaccinated_deceased_individuals
        ]
        self.number_of_fully_vaccinated_deceased_individuals_list = [
            self.environment.number_of_fully_vaccinated_deceased_individuals
        ]
        self.number_of_booster_vaccinated_deceased_individuals_list = [
            self.environment.number_of_booster_vaccinated_deceased_individuals
        ]
        self.number_of_deceased_individuals_list = [
            self.environment.number_of_deceased_individuals
        ]

        # Vaccinated:
        self.number_of_unvaccinated_individuals_list = [
            self.environment.number_of_unvaccinated_individuals
        ]
        self.number_of_fully_vaccinated_individuals_list = [
            self.environment.number_of_fully_vaccinated_individuals
        ]
        self.number_of_booster_vaccinated_individuals_list = [
            self.environment.number_of_booster_vaccinated_individuals
        ]

        self.economic_and_public_perception_rate_list = [
            self.environment.economic_and_public_perception_rate
        ]
        self.number_of_new_infections_list = []
        self.rewards_per_episode_training = []
        self.rewards_per_episode_evaluation = []
        self.action_history = []

        self.population_dynamics = {}

    def build_neural_network(self):
        """This method builds the actor and critic networks."""

        observation_input = Input(
            shape=(self.time_period, self.environment.observation_space.n),
            dtype=tf.float32,
        )
        common1 = LSTM(512, return_sequences=True)(observation_input)
        common2 = LSTM(256, return_sequences=False)(common1)
        # common3 = Dense(units=128, activation=tf.tanh)

        logits = Dense(self.environment.action_space.n)(common2)
        value = Dense(1, activation="linear")(common2)

        actor = Model(inputs=observation_input, outputs=logits)
        critic = Model(inputs=observation_input, outputs=value)

        policy_optimizer = Adam(learning_rate=self.policy_learning_rate)
        value_optimizer = Adam(learning_rate=self.value_function_learning_rate)

        return actor, critic, policy_optimizer, value_optimizer

    def log_probabilities(self, logits, a):
        """This method computes the log-probabilities of taking actions a by using the logits
        (i.e. the output of the actor)"""

        log_probabilities_all = tf.nn.log_softmax(logits)
        log_probability = tf.reduce_sum(
            tf.one_hot(a, self.number_of_actions) * log_probabilities_all, axis=1
        )
        return log_probability

    def sample_action(self, observation, evaluation=False):
        """This method gets the logits from the actor network, converts them to action probabilities, ensures that
        illegal actions have a zero probability, adds noise to the allowed action probabilities and returns the logits
        and action.
        """

        logits = self.actor(observation)

        # Reweighing the action-probabilities to allow only legal actions.
        action_probabilities = tf.nn.softmax(logits)[0].numpy()
        action_probabilities = np.asarray(action_probabilities)
        allowed_actions_numbers = np.asarray(self.environment.allowed_actions_numbers)
        allowed_action_probabilities = allowed_actions_numbers * action_probabilities
        remainder_probability = 1 - np.sum(allowed_action_probabilities)
        allowed_action_probabilities += (
            remainder_probability
            * allowed_actions_numbers
            / np.count_nonzero(allowed_actions_numbers)
        )

        # Adding noise to allowed action probabilities.
        allowed_action_probabilities += (
            1 / self.environment.action_space.n * (self.number_of_epochs + 1)
        ) * allowed_actions_numbers
        allowed_action_probabilities /= np.sum(allowed_action_probabilities)

        # Selecting an action.
        if evaluation:
            action = np.argmax(allowed_action_probabilities)
        else:
            action = np.random.choice(
                self.environment.action_space.n, p=allowed_action_probabilities
            )
        action = tf.constant([action])

        return logits, action

    def train_policy(
        self,
        observation_buffer,
        action_buffer,
        log_probability_buffer,
        advantage_buffer,
    ):
        """This method trains the policy by maximizing the PPO-Clip objective."""

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.log_probabilities(self.actor(observation_buffer), action_buffer)
                - log_probability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_gradients = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(
            zip(policy_gradients, self.actor.trainable_variables)
        )

        kl = tf.reduce_mean(
            log_probability_buffer
            - self.log_probabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)

        return kl

    def train_value_function(self, observation_buffer, return_buffer):
        """This method trains the critic by regression on mean-squared error."""

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean(
                (return_buffer - self.critic(observation_buffer)) ** 2
            )
        value_gradients = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(
            zip(value_gradients, self.critic.trainable_variables)
        )

    def train(self):
        """This method performs the agent training."""

        with tf.device("/device:GPU:0"):
            # Initialize the observation, episode return and episode length
            observation, info = self.environment.reset()
            observation = [observation for _ in range(self.time_period)]
            observation = deque(observation, maxlen=self.time_period)
            episode_return, episode_length = 0, 0

            training_start_time = time.time()
            total_number_of_episodes = 0

            for epoch in range(self.number_of_epochs):
                print("\nEpoch:", epoch + 1)
                epoch_start_time = time.time()
                sample_generation_start_time = time.time()

                # Initialize the sum of the returns, lengths and number of episodes for each epoch.
                sum_return = 0
                sum_length = 0
                number_of_episodes = 0

                # Iterate over the steps of each epoch.
                for t in range(self.steps_per_epoch):
                    # Get the logits, action, and take one step in the environment.
                    logits, action = self.sample_action(
                        np.asarray(observation).reshape(
                            (-1, self.time_period, self.environment.observation_space.n)
                        )
                    )

                    (
                        new_observation,
                        reward,
                        terminated,
                        truncated,
                        _,
                    ) = self.environment.step(action[0].numpy())

                    episode_return += reward
                    episode_length += 1

                    # Get the value and log-probability of the action.
                    value_t = self.critic(
                        np.asarray(observation).reshape(
                            (-1, self.time_period, self.environment.observation_space.n)
                        )
                    )
                    log_probability_t = self.log_probabilities(logits, action)

                    # Store the observation, action, reward, state value, and log_probabilities of actions.
                    self.buffer.store(
                        np.asarray(observation).reshape(
                            (-1, self.time_period, self.environment.observation_space.n)
                        ),
                        action,
                        reward,
                        value_t,
                        log_probability_t,
                    )

                    observation.append(new_observation)

                    # Finish trajectory if a terminal state is reached.
                    terminal = terminated or truncated

                    if terminal or (t == self.steps_per_epoch - 1):
                        last_value = (
                            0
                            if terminal
                            else self.critic(
                                np.asarray(observation).reshape(
                                    (
                                        -1,
                                        self.time_period,
                                        self.environment.observation_space.n,
                                    )
                                )
                            )
                        )
                        self.buffer.finish_trajectory(last_value)
                        sum_return += episode_return
                        sum_length += episode_length
                        number_of_episodes += 1
                        self.rewards_per_episode_training.append(episode_return)

                        observation, info = self.environment.reset()
                        observation = [observation for _ in range(self.time_period)]
                        observation = deque(observation, maxlen=self.time_period)
                        episode_return, episode_length = 0, 0

                # Get values from the buffer
                (
                    observation_buffer,
                    action_buffer,
                    advantage_buffer,
                    return_buffer,
                    log_probability_buffer,
                ) = self.buffer.get()

                sample_generation_end_time = time.time()
                epoch_training_start_time = time.time()

                # Update the policy and implement early stopping using KL divergence.
                for _ in range(self.train_policy_iterations):
                    kl = self.train_policy(
                        observation_buffer,
                        action_buffer,
                        log_probability_buffer,
                        advantage_buffer,
                    )
                    if kl > 1.5 * self.target_kl:
                        # Early Stopping.
                        break

                # Update the value function
                for _ in range(self.train_value_iterations):
                    self.train_value_function(observation_buffer, return_buffer)

                epoch_training_end_time = time.time()

                total_number_of_episodes += number_of_episodes
                mean_return = sum_return / number_of_episodes
                mean_length = sum_length / number_of_episodes

                # Print training statistics.
                print(
                    f"Epoch: {epoch + 1} Number of Episodes: {number_of_episodes}, Mean Return: {mean_return},"
                    f" Mean Length: {mean_length}"
                )
                print(
                    "Sample Generation Time:",
                    sample_generation_end_time - sample_generation_start_time,
                )

                print(
                    "Epoch Training Time:",
                    epoch_training_end_time - epoch_training_start_time,
                )
                print("Epoch Time:", time.time() - epoch_start_time)

            print("Total Number of Episodes:", total_number_of_episodes)
            print("Total Time:", time.time() - training_start_time)

            self.evaluate(epoch=epoch)

    def evaluate(self, epoch):
        """This method evaluates the trained agent's performance."""

        # Initialize the observation, episode return and episode length
        observation, info = self.environment.reset()
        observation = [observation for _ in range(self.time_period)]
        observation = deque(observation, maxlen=self.time_period)
        terminated, truncated = False, False
        terminal = terminated or truncated
        episode_return, episode_length = 0, 0

        # Lists for plotting:
        self.number_of_unvaccinated_susceptible_individuals_list = []
        self.number_of_fully_vaccinated_susceptible_individuals_list = []
        self.number_of_booster_vaccinated_susceptible_individuals_list = []
        self.number_of_susceptible_individuals_list = []

        self.number_of_unvaccinated_exposed_individuals_list = []
        self.number_of_fully_vaccinated_exposed_individuals_list = []
        self.number_of_booster_vaccinated_exposed_individuals_list = []
        self.number_of_exposed_individuals_list = []

        self.number_of_unvaccinated_infected_individuals_list = []
        self.number_of_fully_vaccinated_infected_individuals_list = []
        self.number_of_booster_vaccinated_infected_individuals_list = []
        self.number_of_infected_individuals_list = []

        self.number_of_unvaccinated_hospitalized_individuals_list = []
        self.number_of_fully_vaccinated_hospitalized_individuals_list = []
        self.number_of_booster_vaccinated_hospitalized_individuals_list = []
        self.number_of_hospitalized_individuals_list = []

        self.number_of_unvaccinated_recovered_individuals_list = []
        self.number_of_fully_vaccinated_recovered_individuals_list = []
        self.number_of_booster_vaccinated_recovered_individuals_list = []
        self.number_of_recovered_individuals_list = []

        self.number_of_unvaccinated_deceased_individuals_list = []
        self.number_of_fully_vaccinated_deceased_individuals_list = []
        self.number_of_booster_vaccinated_deceased_individuals_list = []
        self.number_of_deceased_individuals_list = []

        self.number_of_unvaccinated_individuals_list = []
        self.number_of_fully_vaccinated_individuals_list = []
        self.number_of_booster_vaccinated_individuals_list = []

        self.number_of_new_infections_list = []
        self.economic_and_public_perception_rate_list = []
        self.action_history = []

        while not terminal:
            # Get the logits, action, and take one step in the environment.
            logits, action = self.sample_action(
                np.asarray(observation).reshape(
                    (-1, self.time_period, self.environment.observation_space.n)
                ),
                evaluation=True,
            )

            (
                new_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = self.environment.step(action[0].numpy())

            terminal = terminated or truncated
            episode_return += reward
            episode_length += 1

            observation.append(new_observation)

            self.action_history.append(action)

            # Appending the population statistics to their lists for plotting the graph.
            self.number_of_unvaccinated_susceptible_individuals_list.append(
                self.environment.number_of_unvaccinated_susceptible_individuals
            )
            self.number_of_fully_vaccinated_susceptible_individuals_list.append(
                self.environment.number_of_fully_vaccinated_susceptible_individuals
            )
            self.number_of_booster_vaccinated_susceptible_individuals_list.append(
                self.environment.number_of_booster_vaccinated_susceptible_individuals
            )
            self.number_of_susceptible_individuals_list.append(
                self.environment.number_of_susceptible_individuals
            )

            self.number_of_unvaccinated_exposed_individuals_list.append(
                self.environment.number_of_unvaccinated_exposed_individuals
            )
            self.number_of_fully_vaccinated_exposed_individuals_list.append(
                self.environment.number_of_fully_vaccinated_exposed_individuals
            )
            self.number_of_booster_vaccinated_exposed_individuals_list.append(
                self.environment.number_of_booster_vaccinated_exposed_individuals
            )
            self.number_of_exposed_individuals_list.append(
                self.environment.number_of_exposed_individuals
            )

            self.number_of_unvaccinated_infected_individuals_list.append(
                self.environment.number_of_unvaccinated_infected_individuals
            )
            self.number_of_fully_vaccinated_infected_individuals_list.append(
                self.environment.number_of_fully_vaccinated_infected_individuals
            )
            self.number_of_booster_vaccinated_infected_individuals_list.append(
                self.environment.number_of_booster_vaccinated_infected_individuals
            )
            self.number_of_infected_individuals_list.append(
                self.environment.number_of_infected_individuals
            )

            self.number_of_unvaccinated_hospitalized_individuals_list.append(
                self.environment.number_of_unvaccinated_hospitalized_individuals
            )
            self.number_of_fully_vaccinated_hospitalized_individuals_list.append(
                self.environment.number_of_fully_vaccinated_hospitalized_individuals
            )
            self.number_of_booster_vaccinated_hospitalized_individuals_list.append(
                self.environment.number_of_booster_vaccinated_hospitalized_individuals
            )
            self.number_of_hospitalized_individuals_list.append(
                self.environment.number_of_hospitalized_individuals
            )

            self.number_of_unvaccinated_recovered_individuals_list.append(
                self.environment.number_of_unvaccinated_recovered_individuals
            )
            self.number_of_fully_vaccinated_recovered_individuals_list.append(
                self.environment.number_of_fully_vaccinated_recovered_individuals
            )
            self.number_of_booster_vaccinated_recovered_individuals_list.append(
                self.environment.number_of_booster_vaccinated_recovered_individuals
            )
            self.number_of_recovered_individuals_list.append(
                self.environment.number_of_recovered_individuals
            )

            self.number_of_unvaccinated_deceased_individuals_list.append(
                self.environment.number_of_unvaccinated_deceased_individuals
            )
            self.number_of_fully_vaccinated_deceased_individuals_list.append(
                self.environment.number_of_fully_vaccinated_deceased_individuals
            )
            self.number_of_booster_vaccinated_deceased_individuals_list.append(
                self.environment.number_of_booster_vaccinated_deceased_individuals
            )
            self.number_of_deceased_individuals_list.append(
                self.environment.number_of_deceased_individuals
            )

            self.number_of_unvaccinated_individuals_list.append(
                self.environment.number_of_unvaccinated_individuals
            )
            self.number_of_fully_vaccinated_individuals_list.append(
                self.environment.number_of_fully_vaccinated_individuals
            )
            self.number_of_booster_vaccinated_individuals_list.append(
                self.environment.number_of_booster_vaccinated_individuals
            )

            self.economic_and_public_perception_rate_list.append(
                self.environment.economic_and_public_perception_rate
            )

        # Create a dataframe from lists
        df = pd.DataFrame(
            list(
                zip(
                    self.environment.covid_data["date"][214:],
                    self.number_of_unvaccinated_individuals_list,
                    self.number_of_fully_vaccinated_individuals_list,
                    self.number_of_booster_vaccinated_individuals_list,
                    self.environment.new_cases,
                    self.number_of_susceptible_individuals_list,
                    self.number_of_exposed_individuals_list,
                    self.number_of_infected_individuals_list,
                    self.number_of_hospitalized_individuals_list,
                    self.number_of_recovered_individuals_list,
                    self.number_of_deceased_individuals_list,
                    self.number_of_unvaccinated_susceptible_individuals_list,
                    self.number_of_fully_vaccinated_susceptible_individuals_list,
                    self.number_of_booster_vaccinated_susceptible_individuals_list,
                    self.number_of_unvaccinated_exposed_individuals_list,
                    self.number_of_fully_vaccinated_exposed_individuals_list,
                    self.number_of_booster_vaccinated_exposed_individuals_list,
                    self.number_of_unvaccinated_infected_individuals_list,
                    self.number_of_fully_vaccinated_infected_individuals_list,
                    self.number_of_booster_vaccinated_infected_individuals_list,
                    self.number_of_unvaccinated_hospitalized_individuals_list,
                    self.number_of_fully_vaccinated_hospitalized_individuals_list,
                    self.number_of_booster_vaccinated_hospitalized_individuals_list,
                    self.number_of_unvaccinated_recovered_individuals_list,
                    self.number_of_fully_vaccinated_recovered_individuals_list,
                    self.number_of_booster_vaccinated_recovered_individuals_list,
                    self.number_of_unvaccinated_deceased_individuals_list,
                    self.number_of_fully_vaccinated_deceased_individuals_list,
                    self.number_of_booster_vaccinated_deceased_individuals_list,
                    self.economic_and_public_perception_rate_list,
                    self.action_history,
                )
            ),
            columns=[
                "date",
                "unvaccinated_individuals",
                "fully_vaccinated_individuals",
                "booster_vaccinated_individuals",
                "New Cases",
                "Susceptible",
                "Exposed",
                "Infected",
                "Hospitalized",
                "Recovered",
                "Deceased",
                "Susceptible_UV",
                "Susceptible_FV",
                "Susceptible_BV",
                "Exposed_UV",
                "Exposed_FV",
                "Exposed_BV",
                "Infected_UV",
                "Infected_FV",
                "Infected_BV",
                "Hospitalized_UV",
                "Hospitalized_FV",
                "Hospitalized_BV",
                "Recovered_UV",
                "Recovered_FV",
                "Recovered_BV",
                "Deceased_UV",
                "Deceased_FV",
                "Deceased_BV",
                "Economic and Public Perception Rate",
                "Action",
            ],
        )

        df.to_csv(f"./Results/DataFrames/{self.date_and_time}/{epoch}.csv")

        print(
            "Timestep:",
            self.environment.timestep,
            "Number of Susceptible People:",
            self.environment.number_of_susceptible_individuals,
            "Number of Exposed People:",
            self.environment.number_of_exposed_individuals,
            "Number of Infected People:",
            self.environment.number_of_infected_individuals,
            "Number of Hospitalized People:",
            self.environment.number_of_hospitalized_individuals,
            "Number of Recovered People:",
            self.environment.number_of_recovered_individuals,
            "Number of Deceased People:",
            self.environment.number_of_deceased_individuals,
            "EPPR:",
            self.environment.economic_and_public_perception_rate,
        )

        self.population_dynamics[epoch] = [
            self.number_of_susceptible_individuals_list,
            self.number_of_exposed_individuals_list,
            self.number_of_infected_individuals_list,
            self.number_of_hospitalized_individuals_list,
            self.number_of_recovered_individuals_list,
            self.number_of_deceased_individuals_list,
            self.economic_and_public_perception_rate_list,
        ]

        print(
            len(self.population_dynamics[epoch]),
            len(self.population_dynamics[epoch][0]),
        )
        print("population Dynamics:", self.population_dynamics[epoch])
        print(
            "Day 30 Susceptible:",
            self.population_dynamics[epoch][0][29],
            "Day 30 Exposed:",
            self.population_dynamics[epoch][1][29],
            "Day 30 Infected:",
            self.population_dynamics[epoch][2][29],
            "Day 30 Hospitalized:",
            self.population_dynamics[epoch][3][29],
            "Day 30 Recovered:",
            self.population_dynamics[epoch][4][29],
            "Day 30 Deceased:",
            self.population_dynamics[epoch][5][29],
            "Day 30 EPPR:",
            self.population_dynamics[epoch][6][29],
        )

        print(
            "Day 60 Susceptible:",
            self.population_dynamics[epoch][0][59],
            "Day 60 Exposed:",
            self.population_dynamics[epoch][1][59],
            "Day 60 Infected:",
            self.population_dynamics[epoch][2][59],
            "Day 60 Hospitalized:",
            self.population_dynamics[epoch][3][59],
            "Day 60 Recovered:",
            self.population_dynamics[epoch][4][59],
            "Day 60 Deceased:",
            self.population_dynamics[epoch][5][59],
            "Day 60 EPPR:",
            self.population_dynamics[epoch][6][59],
        )

        print(
            "Day 90 Susceptible:",
            self.population_dynamics[epoch][0][89],
            "Day 90 Exposed:",
            self.population_dynamics[epoch][1][89],
            "Day 90 Infected:",
            self.population_dynamics[epoch][2][89],
            "Day 90 Hospitalized:",
            self.population_dynamics[epoch][3][89],
            "Day 90 Recovered:",
            self.population_dynamics[epoch][4][89],
            "Day 90 Deceased:",
            self.population_dynamics[epoch][5][89],
            "Day 90 EPPR:",
            self.population_dynamics[epoch][6][89],
        )

        print(
            "Day 120 Susceptible:",
            self.population_dynamics[epoch][0][119],
            "Day 120 Exposed:",
            self.population_dynamics[epoch][1][119],
            "Day 120 Infected:",
            self.population_dynamics[epoch][2][119],
            "Day 120 Hospitalized:",
            self.population_dynamics[epoch][3][119],
            "Day 120 Recovered:",
            self.population_dynamics[epoch][4][119],
            "Day 120 Deceased:",
            self.population_dynamics[epoch][5][119],
            "Day 120 EPPR:",
            self.population_dynamics[epoch][6][119],
        )

        print(
            "Day 180 Susceptible:",
            self.population_dynamics[epoch][0][179],
            "Day 180 Exposed:",
            self.population_dynamics[epoch][1][179],
            "Day 180 Infected:",
            self.population_dynamics[epoch][2][179],
            "Day 180 Hospitalized:",
            self.population_dynamics[epoch][3][179],
            "Day 180 Recovered:",
            self.population_dynamics[epoch][4][179],
            "Day 180 Deceased:",
            self.population_dynamics[epoch][5][179],
            "Day 180 EPPR:",
            self.population_dynamics[epoch][6][179],
        )

        print("Action History:", self.action_history)
        print("Total Reward:", episode_return, "Episode Length:", episode_length)

    def plots(self):
        """This method plots the reward dynamics."""

        plt.figure(figsize=(20, 10))
        plt.plot(self.rewards_per_episode_training, "ro")
        plt.xlabel("Episodes")
        plt.ylabel("Reward Value")
        plt.title("Rewards Per Episode (During Training)")
        plt.grid()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()


epidemic_simulation_environment = EpidemicSimulation(
    data_path="../../../Data/Updated Data/epidemiological_model_data/",
    state_name="new_york",
    state_population=19_453_734,
    start_date="11/01/2021",
)

ppo_initialization_parameters = {
    "time_period": 14,
    "steps_per_epoch": 10000,
    "number_of_epochs": 50,
    "gamma": 0.999,
    "clip_ratio": 0.2,
    "policy_learning_rate": 1e-3,
    "value_function_learning_rate": 1e-4,
    "train_policy_iteration": 100,
    "train_value_iterations": 100,
    "lam": 0.95,
    "target_kl": 0.01,
}

ppo = ProximalPolicyOptimization(
    environment=epidemic_simulation_environment,
    initialization_parameters=ppo_initialization_parameters,
)

ppo.train()
ppo.plots()
ppo.evaluate(epoch=1)
