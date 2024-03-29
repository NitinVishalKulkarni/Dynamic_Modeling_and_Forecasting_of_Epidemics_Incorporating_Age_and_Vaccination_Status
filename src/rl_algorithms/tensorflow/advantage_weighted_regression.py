# Imports
from collections import deque
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.layers import Dense, Input, LSTM
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
import time
import logging
from datetime import datetime
import pathlib


# This ensures that all the data isn't loaded into the GPU memory at once.
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# Disables eager execution.
tf.compat.v1.disable_eager_execution()

# Disables Tensorflow messages.
logging.getLogger('tensorflow').disabled = True
tf.compat.v1.experimental.output_all_intermediates(True)


# noinspection DuplicatedCode
class AdvantageWeightedRegression:
    """This class implements the AWR Agent."""

    def __init__(self, environment, alternate_network=False, offline_memory_size=10_000, iterations=10):
        """This method initializes the AWR parameters, and calls the train, evaluate and render_actions methods.

        :param environment: This is the environment on which the agent will learn.
        :param alternate_network: Boolean indicating whether to use the second deeper network.
        :param offline_memory_size: Integer indicating the size of the offline replay memory.
        :param iterations: Integer indicating the number of iterations for which the agent will train."""

        # Saving the training results.
        self.date_and_time = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        pathlib.Path(f'./Results/DataFrames/{self.date_and_time}').mkdir(parents=True, exist_ok=True)

        self.environment = environment  # The environment which we need the agent to solve.
        self.environment.reset()
        self.alternate_network = alternate_network  # Boolean indicating whether to use the second deeper network.
        self.offline_replay_memory_size = offline_memory_size  # This specifies the size of the offline replay memory.
        # self.offline_replay_memory = []  # Offline replay memory.
        self.offline_replay_memory = deque(maxlen=self.offline_replay_memory_size)
        self.iterations = iterations  # Number of episodes for which the agent will train.
        self.discount_factor = 0.999  # Discount factor determines the value of the future rewards.
        self.beta = 0.5  # Hyperparameter used to calculate the exponential advantage.
        self.time_period = 14  # Number of days to consider when taking an action.
        self.actor_model, self.critic_model, self.policy_model = self.neural_network()  # Creating the networks.
        self.cumulative_rewards_evaluation = []  # List containing the cumulative rewards per episode during evaluation.

        # print('NN Summary:', self.actor_model.summary(), self.critic_model.summary(), self.policy_model.summary())
        # sys.exit()

        # Lists for plotting:
        self.number_of_unvaccinated_susceptible_individuals_list = \
            [self.environment.number_of_unvaccinated_susceptible_individuals]
        self.number_of_fully_vaccinated_susceptible_individuals_list = \
            [self.environment.number_of_fully_vaccinated_susceptible_individuals]
        self.number_of_booster_vaccinated_susceptible_individuals_list = \
            [self.environment.number_of_booster_vaccinated_susceptible_individuals]
        self.number_of_susceptible_individuals_list = [self.environment.number_of_susceptible_individuals]

        self.number_of_unvaccinated_exposed_individuals_list = \
            [self.environment.number_of_unvaccinated_exposed_individuals]
        self.number_of_fully_vaccinated_exposed_individuals_list = \
            [self.environment.number_of_fully_vaccinated_exposed_individuals]
        self.number_of_booster_vaccinated_exposed_individuals_list = \
            [self.environment.number_of_booster_vaccinated_exposed_individuals]
        self.number_of_exposed_individuals_list = \
            [self.environment.number_of_exposed_individuals]

        self.number_of_unvaccinated_infected_individuals_list = \
            [self.environment.number_of_unvaccinated_infected_individuals]
        self.number_of_fully_vaccinated_infected_individuals_list = \
            [self.environment.number_of_fully_vaccinated_infected_individuals]
        self.number_of_booster_vaccinated_infected_individuals_list = \
            [self.environment.number_of_booster_vaccinated_infected_individuals]
        self.number_of_infected_individuals_list = \
            [self.environment.number_of_infected_individuals]

        self.number_of_unvaccinated_hospitalized_individuals_list = \
            [self.environment.number_of_unvaccinated_hospitalized_individuals]
        self.number_of_fully_vaccinated_hospitalized_individuals_list = \
            [self.environment.number_of_fully_vaccinated_hospitalized_individuals]
        self.number_of_booster_vaccinated_hospitalized_individuals_list = \
            [self.environment.number_of_booster_vaccinated_hospitalized_individuals]
        self.number_of_hospitalized_individuals_list = \
            [self.environment.number_of_hospitalized_individuals]

        self.number_of_unvaccinated_recovered_individuals_list = \
            [self.environment.number_of_unvaccinated_recovered_individuals]
        self.number_of_fully_vaccinated_recovered_individuals_list = \
            [self.environment.number_of_fully_vaccinated_recovered_individuals]
        self.number_of_booster_vaccinated_recovered_individuals_list = \
            [self.environment.number_of_booster_vaccinated_recovered_individuals]
        self.number_of_recovered_individuals_list = \
            [self.environment.number_of_recovered_individuals]

        self.number_of_unvaccinated_deceased_individuals_list = \
            [self.environment.number_of_unvaccinated_deceased_individuals]
        self.number_of_fully_vaccinated_deceased_individuals_list = \
            [self.environment.number_of_fully_vaccinated_deceased_individuals]
        self.number_of_booster_vaccinated_deceased_individuals_list = \
            [self.environment.number_of_booster_vaccinated_deceased_individuals]
        self.number_of_deceased_individuals_list = \
            [self.environment.number_of_deceased_individuals]

        self.number_of_unvaccinated_individuals_list = \
            [self.environment.number_of_unvaccinated_individuals]
        self.number_of_fully_vaccinated_individuals_list = \
            [self.environment.number_of_fully_vaccinated_individuals]
        self.number_of_booster_vaccinated_individuals_list = \
            [self.environment.number_of_booster_vaccinated_individuals]

        self.population_dynamics = {}
        self.economic_and_social_rate_list = [self.environment.economic_and_public_perception_rate]
        self.action_history = []
        # self.train()  # Calling the train method.
        # self.evaluate()  # Calling the evaluate method.
        self.render_actions(1)  # Calling the render method.

    def neural_network(self):
        """This method builds the actor, critic and policy networks."""

        if not self.alternate_network:
            # Input 1 is the one-hot representation of the environment state.
            input_ = Input(shape=(self.time_period, self.environment.observation_space.n))
            # Input 2 is the exponential advantage.
            exponential_advantage = Input(shape=[1])
            common1 = LSTM(512, return_sequences=True)(input_)  # Common layer 1 for the networks.
            common2 = LSTM(256, return_sequences=False)(common1)
            probabilities = Dense(self.environment.action_space.n, activation='softmax')(common2)  # Actor output.
            values = Dense(1, activation='linear')(common2)  # Critic output.

        else:
            # Input 1 is the one-hot representation of the environment state.
            input_ = Input(shape=(self.environment.observation_space.n,))
            # Input 2 is the exponential advantage.
            exponential_advantage = Input(shape=[1])
            common1 = Dense(1024, activation='relu')(input_)  # Common layer 1 for the networks.
            common2 = Dense(512, activation='relu')(common1)  # Common layer 2 for the networks.
            common3 = Dense(256, activation='relu')(common2)  # Common layer 3 for the networks.
            probabilities = Dense(self.environment.action_space.n, activation='softmax')(common3)  # Actor output.
            values = Dense(1, activation='linear')(common3)  # Critic output.

        def custom_loss(exponential_advantage_):
            """This method defines the custom loss wrapper function that will be used by the actor model."""

            def loss_fn(y_true, y_pred):
                # Clipping y_pred so that we don't end up taking the log of 0 or 1.
                clipped_y_pred = k.clip(y_pred, 1e-8, 1 - 1e-8)
                log_probability = y_true * k.log(clipped_y_pred)
                return k.sum(-log_probability * exponential_advantage_)

            return loss_fn

        lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.96)

        # Instantiating the actor model.
        actor_model = Model(inputs=[input_, exponential_advantage], outputs=[probabilities])
        actor_model.compile(optimizer=Adam(), loss=custom_loss(exponential_advantage))
        # actor_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=custom_loss(exponential_advantage))

        # Instantiating the critic model.
        critic_model = Model(inputs=[input_], outputs=[values])
        critic_model.compile(optimizer=Adam(), loss=tf.keras.losses.Huber())

        # Instantiating the policy model.
        policy_model = Model(inputs=[input_], outputs=[probabilities])

        return actor_model, critic_model, policy_model

    def monte_carlo_returns(self):
        """This method calculates the Monte Carlo returns given a list of rewards."""

        rewards = [item[2] for item in self.offline_replay_memory]
        monte_carlo_returns = []  # List containing the Monte-Carlo returns.
        monte_carlo_return = 0
        t = 0  # Exponent by which the discount factor is raised.

        for i in range(len(self.offline_replay_memory)):

            while not self.offline_replay_memory[i][4]:  # Execute until you encounter a terminal state.

                # Equation to calculate the Monte-Carlo return.
                monte_carlo_return += self.discount_factor ** t * rewards[i]
                i += 1  # Go to the next sample.
                t += 1  # Increasing the exponent by which the discount factor is raised.

                # Condition to check whether we have reached the end of the replay memory without the episode being
                # terminated, and if so break. (This can happen with the samples at the end of the replay memory as we
                # only store the samples till we reach the replay memory size and not till we exceed it with the episode
                # being terminated.)
                if i == len(self.offline_replay_memory):
                    # If the episode hasn't terminated but you reach the end append the Monte-Carlo return to the list.
                    monte_carlo_returns.append(monte_carlo_return)

                    # Resetting the Monte-Carlo return value and the exponent to 0.
                    monte_carlo_return = 0
                    t = 0

                    break  # Break from the loop.

            # If for one of the samples towards the end we reach the end of the replay memory and it hasn't terminated,
            # we will go back to the beginning of the for loop to calculate the Monte-Carlo return for the future
            # samples if any for whom the episode hasn't terminated.
            if i == len(self.offline_replay_memory):
                continue

            # Equation to calculate the Monte-Carlo return.
            monte_carlo_return += self.discount_factor ** t * rewards[i]

            # Appending the Monte-Carlo Return for cases where the episode terminates without reaching the end of the
            # replay memory.
            monte_carlo_returns.append(monte_carlo_return)

            # Resetting the Monte-Carlo return value and the exponent to 0.
            monte_carlo_return = 0
            t = 0

        # Normalizing the returns.
        monte_carlo_returns = np.array(monte_carlo_returns)
        monte_carlo_returns = (monte_carlo_returns - np.mean(monte_carlo_returns)) / (np.std(monte_carlo_returns)
                                                                                      + 1e-08)
        monte_carlo_returns = monte_carlo_returns.tolist()

        return monte_carlo_returns

    def td_lambda_returns(self):
        """This method calculates the TD Lambda returns."""

        rewards = [item[2] for item in self.offline_replay_memory]
        next_states = [item[3] for item in self.offline_replay_memory]
        next_states = np.asarray(next_states).reshape(-1, self.environment.observation_space.n)
        next_state_values = self.critic_model.predict(next_states).flatten()
        td_lambda_returns = []  # List containing the TD Lambda returns.
        terminal_state_indices = [i for i in range(len(self.offline_replay_memory)) if self.offline_replay_memory[i][4]]
        td_n_return = 0
        t = 0  # Exponent by which the discount factor is raised.
        td_lambda_value = 0.9
        index = 0  # Pointer for keeping track of the next terminal state.
        next_terminal_state_index = terminal_state_indices[index]
        for i in range(len(self.offline_replay_memory)):
            j = i  # Used to calcuate the lambda values by which we will multiply the TD (n) returns.
            if i > terminal_state_indices[index] and index < len(terminal_state_indices) - 1:
                index += 1
                next_terminal_state_index = terminal_state_indices[index]
            td_n_returns = []  # List containing the TD (n) returns.
            xyz = 0
            for n in range(next_terminal_state_index, next_terminal_state_index + 1):
                while i != n:  # Execute until you encounter a terminal state.

                    # Equation to calculate the Monte-Carlo return.
                    td_n_return += self.discount_factor ** t * rewards[i]

                    td_n_returns.append(td_n_return + self.discount_factor ** (t + 1) * next_state_values[i])
                    i += 1  # Go to the next sample.
                    t += 1  # Increasing the exponent by which the discount factor is raised.

                    # Condition to check whether we have reached the end of the replay memory without the episode being
                    # terminated, and if so break. (This can happen with the samples at the end of the replay memory as
                    # we only store the samples till we reach the replay memory size and not till we exceed it with the
                    # episode being terminated.)
                    if i == len(self.offline_replay_memory):
                        # Resetting the Monte-Carlo return value and the exponent to 0.
                        td_n_return = 0
                        t = 0

                        break  # Break from the loop.

                # If for one of the samples towards the end we reach the end of the replay memory and it hasn't
                # terminated, we will go back to the beginning of the for loop to calculate the Monte-Carlo return for
                # the future samples if any for whom the episode hasn't terminated.
                if i == len(self.offline_replay_memory):
                    continue

                # Equation to calculate the Monte-Carlo return.
                td_n_return += self.discount_factor ** t * rewards[i]
                td_n_return += self.discount_factor ** (t + 1) * next_state_values[i]
                # Appending the Monte-Carlo Return for cases where the episode terminates without reaching the end of
                # the replay memory.
                td_n_returns.append(td_n_return)

                # Resetting the Monte-Carlo return value and the exponent to 0.
                td_n_return = 0
                t = 0
            if i > terminal_state_indices[index] and index == len(terminal_state_indices) - 1:
                xyz = len(self.offline_replay_memory) - next_terminal_state_index - 1
            values_to_multiply = [td_lambda_value ** x for x in range(next_terminal_state_index + 1 - j + xyz)]
            td_lambda_returns.append((1 - td_lambda_value) * np.dot(values_to_multiply, td_n_returns))

        # Normalizing the returns.
        td_lambda_returns = np.array(td_lambda_returns)
        td_lambda_returns = (td_lambda_returns - np.mean(td_lambda_returns)) / (np.std(td_lambda_returns)
                                                                                + 1e-08)
        td_lambda_returns = td_lambda_returns.tolist()

        return td_lambda_returns

    def replay(self):
        """This is the replay method, that is used to fit the actor and critic networks and synchronize the weights
            between the actor and policy networks."""

        states = [item[0] for item in self.offline_replay_memory]
        states = np.asarray(states).reshape((-1, self.time_period, self.environment.observation_space.n))

        actions = [tf.keras.utils.to_categorical(item[1], self.environment.action_space.n).tolist()
                   for item in self.offline_replay_memory]

        monte_carlo_returns = self.monte_carlo_returns()

        critic_values = self.critic_model.predict(states).flatten()

        # exponential_advantages = [np.exp(1/self.beta * (monte_carlo_returns[i] - critic_values[i]))
        #               for i in range(len(self.offline_replay_memory))]

        advantages = [monte_carlo_returns[i] - critic_values[i]
                      for i in range(len(self.offline_replay_memory))]

        # advantages = [monte_carlo_returns[i] - critic_values[i]
        #               for i in range(len(states))]

        # Fitting the actor model.
        self.actor_model.fit([states, np.asarray(advantages)], np.asarray(actions),
                             batch_size=256, epochs=5, verbose=0)

        # Syncing the weights between the actor and policy models.
        self.policy_model.set_weights(self.actor_model.get_weights())

        # Fitting the critic model.
        self.critic_model.fit(states, np.asarray(monte_carlo_returns), batch_size=256, epochs=5, verbose=0)

    def train(self):
        """This method performs the agent training."""

        average_reward_per_episode_per_iteration = []
        cumulative_average_rewards_per_episode_per_iteration = []

        for iteration in range(self.iterations):
            start = time.time()
            print(f'\n\n Iteration {iteration + 1}')

            # self.offline_replay_memory = []  # Resetting the offline replay memory to be empty.
            total_reward_iteration = 0  # Total reward acquired in this iteration.
            episodes = 0  # Initializing the number of episodes in this iteration to be 0.

            for _ in range(100):
            # while len(self.offline_replay_memory) < self.offline_replay_memory_size:

                # Resetting the environment and starting from a random position.
                state = self.environment.reset()
                state = [state for _ in range(self.time_period)]
                state = deque(state, maxlen=self.time_period)

                done = False  # Initializing the done parameter which indicates whether the environment has terminated
                # or not to False.
                episodes += 1  # Increasing the number of episodes in this iteration.

                while not done:
                    # Selecting an action according to the predicted action probabilities.
                    action_probabilities = (self.policy_model.predict(np.asarray(state).reshape(
                        (-1, self.time_period, self.environment.observation_space.n)))[0])

                    # Adding noise to do exploration and making only legal actions available in a state.
                    # print('\nPrevious Action:', self.environment.previous_action)
                    # print('Current Action:', self.environment.current_action)
                    # print('Action History:', self.environment.action_history, len(self.environment.action_history))
                    # if self.environment.action_history[-16:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                    #     print('omg')
                    #     print('\nPrevious Action:', self.environment.previous_action)
                    #     print('Current Action:', self.environment.current_action)
                    #     print('Action History:', self.environment.action_history, len(self.environment.action_history))
                    #     print('Original Probabilities:', action_probabilities)
                    #     print('Allowed Actions:', self.environment.allowed_actions)
                    #     print('Required Actions:', self.environment.required_actions)
                    #     print('Allowed Actions One Hot:', np.asarray(self.environment.allowed_actions_numbers))

                    # print('Original Probabilities:', action_probabilities)
                    action_probabilities = np.asarray(action_probabilities)
                    allowed_actions_numbers = np.asarray(self.environment.allowed_actions_numbers)
                    allowed_action_probabilities = allowed_actions_numbers * action_probabilities
                    remainder_probability = 1 - np.sum(allowed_action_probabilities)
                    allowed_action_probabilities += \
                        (remainder_probability * allowed_actions_numbers
                         / np.count_nonzero(allowed_actions_numbers))
                    # print('Allowed Actions:', self.environment.allowed_actions)
                    # print('Required Actions:', self.environment.required_actions)
                    # print('Allowed Actions One Hot:', allowed_actions_numbers)
                    # print('Before Noise Allowed Action Probabilities:', allowed_action_probabilities)

                    """Adding noise only to allowed actions:"""
                    allowed_action_probabilities += ((1 / self.environment.action_space.n * (iteration + 1))
                                                     * allowed_actions_numbers)
                    # allowed_action_probabilities += (0.025 * allowed_actions_numbers)
                    allowed_action_probabilities /= np.sum(allowed_action_probabilities)
                    # print('After Noise Allowed Action Probabilities:', allowed_action_probabilities)

                    # DEBUGGING TEST
                    # if len(self.environment.action_history) > 0:
                    #     if self.environment.action_history[-1] == 0:
                    #         print(self.environment.action_history)
                    #         print('Action 0 probabilities:', self.environment.timestep, allowed_action_probabilities)

                    """Adding noise to all actions:"""
                    # action_probabilities += 1 / self.environment.action_space.n * (iteration + 1)
                    # allowed_action_probabilities += 0.025
                    # allowed_action_probabilities /= np.sum(allowed_action_probabilities)

                    # action_probabilities += 0.025
                    # action_probabilities /= np.sum(action_probabilities)
                    # print('New Probabilities:', allowed_action_probabilities)

                    action = np.random.choice(self.environment.action_space.n, p=allowed_action_probabilities)

                    # Taking an action.
                    next_state, reward, done, info = self.environment.step(action)

                    # Incrementing the total reward.
                    total_reward_iteration += reward

                    # Appending the state, action, reward, next state and done to the replay memory.
                    self.offline_replay_memory.append([state, action, reward, next_state, done])

                    # state = next_state  # Setting the current state to be equal to the next state.
                    state.append(next_state)

                    # # This condition ensures that we don't append more values than the size of the replay memory.
                    # if len(self.offline_replay_memory) == self.offline_replay_memory_size:
                    #     break

            # Calculating the average reward per episode for this iteration.
            average_reward_per_episode = total_reward_iteration / episodes
            average_reward_per_episode_per_iteration.append(average_reward_per_episode)

            # Appending the cumulative reward.
            if len(cumulative_average_rewards_per_episode_per_iteration) == 0:
                cumulative_average_rewards_per_episode_per_iteration.append(average_reward_per_episode)
            else:
                cumulative_average_rewards_per_episode_per_iteration.append(
                    average_reward_per_episode + cumulative_average_rewards_per_episode_per_iteration[iteration - 1])

            print('Time to generate samples:', time.time() - start)
            print('Length of Replay Memory:', len(self.offline_replay_memory))

            # Calling the replay method.
            start = time.time()
            self.replay()
            print('Time to train:', time.time() - start)

            self.render_actions(iteration + 1)

        # Calling the plots method to plot the reward dynamics.
        # self.plots(average_reward_per_episode_per_iteration, cumulative_average_rewards_per_episode_per_iteration,
        #            iterations=True)

    def evaluate(self):
        """This method evaluates the performance of the agent after it has finished training."""

        total_steps = 0  # Initializing the total steps taken and total penalties incurred
        # across all episodes.
        episodes = 100  # Number of episodes for which we are going to test the agent's performance.
        rewards_per_episode = []  # Sum of immediate rewards during the episode.
        # gold = 0  # Counter to keep track of the episodes in which the agent reaches the Gold.

        for episode in range(episodes):
            state = self.environment.reset()  # Resetting the environment for every new episode.
            steps = 0  # Initializing the steps taken, and penalties incurred in this episode.
            done = False  # Initializing the done parameter indicating the episode termination to be False.
            total_reward_episode = 0  # Initializing the total reward acquired in this episode to be 0.

            while not done:
                # Always choosing the greedy action.
                action = np.argmax(self.policy_model.predict(
                    np.asarray(state).reshape(-1, self.environment.observation_space.n))[0])

                # Taking the greedy action.
                next_state, reward, done, info = self.environment.step(action)

                total_reward_episode += reward  # Adding the reward acquired on this step to the total reward acquired
                # during the episode.

                state = next_state  # Setting the current state to the next state.

                steps += 1  # Increasing the number of steps taken in this episode.

            rewards_per_episode.append(total_reward_episode)  # Appending the reward acquired during the episode.

            # Appending the cumulative reward.
            if len(self.cumulative_rewards_evaluation) == 0:
                self.cumulative_rewards_evaluation.append(total_reward_episode)
            else:
                self.cumulative_rewards_evaluation.append(
                    total_reward_episode + self.cumulative_rewards_evaluation[episode - 1])

            total_steps += steps  # Adding the steps taken in this episode to the total steps taken across all episodes

        # Printing some statistics after the evaluation of agent's performance is completed.
        print(f"\nEvaluation of agent's performance across {episodes} episodes:")
        print(f"Average number of steps taken per episode: {total_steps / episodes}\n")

        # Calling the plots method to plot the reward dynamics.
        # self.plots(rewards_per_episode, self.cumulative_rewards_evaluation)

    def render_actions(self, iteration):
        # Rendering the actions taken by the agent after learning.
        state = self.environment.reset()  # Resetting the environment for a new episode.
        state = [state for _ in range(self.time_period)]
        state = deque(state, maxlen=self.time_period)
        done = False  # Initializing the done parameter indicating the episode termination to be False.

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

        self.new_cases_list = []

        self.economic_and_social_rate_list = []
        self.action_history = []

        while not done:
            # Always choosing the greedy action.
            # action = np.argmax(self.policy_model.predict(
            #     np.asarray(state).reshape((-1, self.time_period, self.environment.observation_space.n)))[0])

            action_probabilities = (self.policy_model.predict(np.asarray(state).reshape(
                (-1, self.time_period, self.environment.observation_space.n)))[0])

            # print('Original Probabilities:', action_probabilities)
            action_probabilities = np.asarray(action_probabilities)
            allowed_actions_numbers = np.asarray(self.environment.allowed_actions_numbers)
            allowed_action_probabilities = allowed_actions_numbers * action_probabilities
            remainder_probability = 1 - np.sum(allowed_action_probabilities)
            allowed_action_probabilities += \
                (remainder_probability * allowed_actions_numbers
                 / np.count_nonzero(allowed_actions_numbers))

            # print('\nPrevious Action:', self.environment.previous_action)
            # print('Current Action:', self.environment.current_action)
            # print('Action History:', self.environment.action_history, len(self.environment.action_history))
            # print('Original Probabilities:', action_probabilities)
            # print('Allowed Actions:', self.environment.allowed_actions)
            # print('Required Actions:', self.environment.required_actions)
            # print('Allowed Actions One Hot:', np.asarray(self.environment.allowed_actions_numbers))
            # print('Before Noise Allowed Action Probabilities:', allowed_action_probabilities)

            # # allowed_action_probabilities += ((1 / self.environment.action_space.n * iteration)
            # #                                  * self.environment.allowed_actions_numbers)
            # allowed_action_probabilities += (0.025 * allowed_actions_numbers)
            # allowed_action_probabilities /= np.sum(allowed_action_probabilities)

            action = np.argmax(allowed_action_probabilities)

            self.action_history.append(action)

            # Taking the greedy action.
            next_state, reward, done, info = self.environment.step(action)

            # Appending the population statistics to their lists for plotting the graph.
            self.number_of_unvaccinated_susceptible_individuals_list.append(
                self.environment.number_of_unvaccinated_susceptible_individuals)
            self.number_of_fully_vaccinated_susceptible_individuals_list.append(
                self.environment.number_of_fully_vaccinated_susceptible_individuals)
            self.number_of_booster_vaccinated_susceptible_individuals_list.append(
                self.environment.number_of_booster_vaccinated_susceptible_individuals)
            self.number_of_susceptible_individuals_list.append(self.environment.number_of_susceptible_individuals)

            self.number_of_unvaccinated_exposed_individuals_list.append(
                self.environment.number_of_unvaccinated_exposed_individuals)
            self.number_of_fully_vaccinated_exposed_individuals_list.append(
                self.environment.number_of_fully_vaccinated_exposed_individuals)
            self.number_of_booster_vaccinated_exposed_individuals_list.append(
                self.environment.number_of_booster_vaccinated_exposed_individuals)
            self.number_of_exposed_individuals_list.append(self.environment.number_of_exposed_individuals)

            self.number_of_unvaccinated_infected_individuals_list.append(
                self.environment.number_of_unvaccinated_infected_individuals)
            self.number_of_fully_vaccinated_infected_individuals_list.append(
                self.environment.number_of_fully_vaccinated_infected_individuals)
            self.number_of_booster_vaccinated_infected_individuals_list.append(
                self.environment.number_of_booster_vaccinated_infected_individuals)
            self.number_of_infected_individuals_list.append(self.environment.number_of_infected_individuals)

            self.number_of_unvaccinated_hospitalized_individuals_list.append(
                self.environment.number_of_unvaccinated_hospitalized_individuals)
            self.number_of_fully_vaccinated_hospitalized_individuals_list.append(
                self.environment.number_of_fully_vaccinated_hospitalized_individuals)
            self.number_of_booster_vaccinated_hospitalized_individuals_list.append(
                self.environment.number_of_booster_vaccinated_hospitalized_individuals)
            self.number_of_hospitalized_individuals_list.append(self.environment.number_of_hospitalized_individuals)

            self.number_of_unvaccinated_recovered_individuals_list.append(
                self.environment.number_of_unvaccinated_recovered_individuals)
            self.number_of_fully_vaccinated_recovered_individuals_list.append(
                self.environment.number_of_fully_vaccinated_recovered_individuals)
            self.number_of_booster_vaccinated_recovered_individuals_list.append(
                self.environment.number_of_booster_vaccinated_recovered_individuals)
            self.number_of_recovered_individuals_list.append(self.environment.number_of_recovered_individuals)

            self.number_of_unvaccinated_deceased_individuals_list.append(
                self.environment.number_of_unvaccinated_deceased_individuals)
            self.number_of_fully_vaccinated_deceased_individuals_list.append(
                self.environment.number_of_fully_vaccinated_deceased_individuals)
            self.number_of_booster_vaccinated_deceased_individuals_list.append(
                self.environment.number_of_booster_vaccinated_deceased_individuals)
            self.number_of_deceased_individuals_list.append(self.environment.number_of_deceased_individuals)

            self.number_of_unvaccinated_individuals_list.append(
                self.environment.number_of_unvaccinated_individuals)
            self.number_of_fully_vaccinated_individuals_list.append(
                self.environment.number_of_fully_vaccinated_individuals)
            self.number_of_booster_vaccinated_individuals_list.append(
                self.environment.number_of_booster_vaccinated_individuals)

            self.economic_and_social_rate_list.append(self.environment.economic_and_public_perception_rate)

            # state = next_state  # Setting the current state to the next state.
            state.append(next_state)

        # Create a dataframe from lists
        df = pd.DataFrame(
            list(zip(self.environment.covid_data['date'][214:],
                     self.number_of_unvaccinated_individuals_list, self.number_of_fully_vaccinated_individuals_list,
                     self.number_of_booster_vaccinated_individuals_list,
                     self.environment.new_cases,
                     self.number_of_susceptible_individuals_list,
                     self.number_of_exposed_individuals_list, self.number_of_infected_individuals_list,
                     self.number_of_hospitalized_individuals_list, self.number_of_recovered_individuals_list,
                     self.number_of_deceased_individuals_list, self.number_of_unvaccinated_susceptible_individuals_list,
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
                     self.economic_and_social_rate_list,
                     self.action_history)),
            columns=['date', 'unvaccinated_individuals', 'fully_vaccinated_individuals',
                     'booster_vaccinated_individuals', 'New Cases',
                     'Susceptible', 'Exposed', 'Infected', 'Hospitalized', 'Recovered', 'Deceased',
                     'Susceptible_UV', 'Susceptible_FV', 'Susceptible_BV', 'Exposed_UV', 'Exposed_FV', 'Exposed_BV',
                     'Infected_UV', 'Infected_FV', 'Infected_BV',
                     'Hospitalized_UV', 'Hospitalized_FV', 'Hospitalized_BV',
                     'Recovered_UV', 'Recovered_FV', 'Recovered_BV', 'Deceased_UV',	'Deceased_FV', 'Deceased_BV',
                     'Economic and Social Perception Rate', 'Action'])

        df.to_csv(f'./Results/DataFrames/{self.date_and_time}/{iteration}.csv')

        print('Timestep:', self.environment.timestep,
              'Number of Susceptible People:', self.environment.number_of_susceptible_individuals,
              'Number of Exposed People:', self.environment.number_of_exposed_individuals,
              'Number of Infected People:', self.environment.number_of_infected_individuals,
              'Number of Hospitalized People:', self.environment.number_of_hospitalized_individuals,
              'Number of Recovered People:', self.environment.number_of_recovered_individuals,
              'Number of Deceased People:', self.environment.number_of_deceased_individuals,
              'GDP:', self.environment.economic_and_public_perception_rate)

        self.population_dynamics[iteration] = \
            [self.number_of_susceptible_individuals_list, self.number_of_exposed_individuals_list,
             self.number_of_infected_individuals_list, self.number_of_hospitalized_individuals_list,
             self.number_of_recovered_individuals_list, self.number_of_deceased_individuals_list,
             self.economic_and_social_rate_list]

        print(len(self.population_dynamics[iteration]), len(self.population_dynamics[iteration][0]))
        print('population Dynamics:', self.population_dynamics[iteration])
        print('Day 30 Susceptible:', self.population_dynamics[iteration][0][29],
              'Day 30 Exposed:', self.population_dynamics[iteration][1][29],
              'Day 30 Infected:', self.population_dynamics[iteration][2][29],
              'Day 30 Hospitalized:', self.population_dynamics[iteration][3][29],
              'Day 30 Recovered:', self.population_dynamics[iteration][4][29],
              'Day 30 Deceased:', self.population_dynamics[iteration][5][29],
              'Day 30 ESR:', self.population_dynamics[iteration][6][29])

        print('Day 60 Susceptible:', self.population_dynamics[iteration][0][59],
              'Day 60 Exposed:', self.population_dynamics[iteration][1][59],
              'Day 60 Infected:', self.population_dynamics[iteration][2][59],
              'Day 60 Hospitalized:', self.population_dynamics[iteration][3][59],
              'Day 60 Recovered:', self.population_dynamics[iteration][4][59],
              'Day 60 Deceased:', self.population_dynamics[iteration][5][59],
              'Day 60 ESR:', self.population_dynamics[iteration][6][59])

        print('Day 90 Susceptible:', self.population_dynamics[iteration][0][89],
              'Day 90 Exposed:', self.population_dynamics[iteration][1][89],
              'Day 90 Infected:', self.population_dynamics[iteration][2][89],
              'Day 90 Hospitalized:', self.population_dynamics[iteration][3][89],
              'Day 90 Recovered:', self.population_dynamics[iteration][4][89],
              'Day 90 Deceased:', self.population_dynamics[iteration][5][89],
              'Day 90 ESR:', self.population_dynamics[iteration][6][89])

        print('Day 120 Susceptible:', self.population_dynamics[iteration][0][119],
              'Day 120 Exposed:', self.population_dynamics[iteration][1][119],
              'Day 120 Infected:', self.population_dynamics[iteration][2][119],
              'Day 120 Hospitalized:', self.population_dynamics[iteration][3][119],
              'Day 120 Recovered:', self.population_dynamics[iteration][4][119],
              'Day 120 Deceased:', self.population_dynamics[iteration][5][119],
              'Day 120 ESR:', self.population_dynamics[iteration][6][119])

        print('Day 180 Susceptible:', self.population_dynamics[iteration][0][179],
              'Day 180 Exposed:', self.population_dynamics[iteration][1][179],
              'Day 180 Infected:', self.population_dynamics[iteration][2][179],
              'Day 180 Hospitalized:', self.population_dynamics[iteration][3][179],
              'Day 180 Recovered:', self.population_dynamics[iteration][4][179],
              'Day 180 Deceased:', self.population_dynamics[iteration][5][179],
              'Day 180 ESR:', self.population_dynamics[iteration][6][179])

        print('Action History:', self.action_history)

        # self.environment.render()  # Rendering the environment.

    @staticmethod
    def plots(rewards_per_episode, cumulative_rewards, iterations=False):
        """This method plots the reward dynamics and epsilon decay.

        :param iterations: Boolean indicating that we are plotting for iterations and not episodes.
        :param rewards_per_episode: List containing the reward values per episode.
        :param cumulative_rewards: List containing the cumulative reward values per episode."""

        plt.figure(figsize=(20, 10))
        plt.plot(rewards_per_episode, 'ro')
        if iterations:
            plt.xlabel('Iterations')
            plt.ylabel('Average Reward Per Episode')
            plt.title('Average Rewards Per Episode Per Iteration')
        else:
            plt.xlabel('Episodes')
            plt.ylabel('Reward Value')
            plt.title('Rewards Per Episode (During Evaluation)')
        plt.grid()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.plot(cumulative_rewards)
        if iterations:
            plt.xlabel('Iterations')
            plt.ylabel('Cumulative Average Reward Per Episode')
            plt.title('Cumulative Average Rewards Per Episode Per Iteration')
        else:
            plt.xlabel('Episodes')
            plt.ylabel('Cumulative Reward Per Episode')
            plt.title('Cumulative Rewards Per Episode (During Evaluation)')
        plt.grid()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()


