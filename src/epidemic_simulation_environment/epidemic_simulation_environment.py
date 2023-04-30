# Imports
from typing import Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.settings import DATA_DIR
from parameter_initializer import ParameterInitializer
from population_dynamics_computer import PopulationDynamicsComputer

pd.set_option("display.max_columns", 50)


# Defining the Epidemic Simulation Environment.
# noinspection DuplicatedCode
class EpidemicSimulationMA(gym.Env):
    """This class implements the Disease Mitigation environment."""

    def __init__(self, env_config):
        """This method initializes the environment parameters.

        :param env_config: Dictionary containing the configuration for environment initialization.
        """

        self.environment_config = env_config
        self.observation_space = spaces.Box(low=0, high=11, shape=(4,))
        self.action_space = spaces.Discrete(12)

        self.parameter_initializer = ParameterInitializer(
            data_path=env_config["data_path"],
            simulation_start_date=env_config["simulation_start_date"],
        )

        self.states = self.parameter_initializer.initialize_state_names()
        # print("State Names:", self.states)

        self.epidemiological_model_data = (
            self.parameter_initializer.initialize_epidemiological_model_data()
        )
        # print("Epidemiological Model Data:\n", self.epidemiological_model_data)

        self.population_dynamics_dataframes = (
            self.parameter_initializer.initialize_population_dynamics()
        )
        # print("\nPopulation Dynamics:\n", self.population_dynamics_dataframes)

        self.epidemiological_model_parameters = (
            self.parameter_initializer.initialize_epidemiological_model_parameters()
        )
        # print("\nEpidemiological Model Parameters:\n", self.epidemiological_model_parameters)

        self.state_populations = (
            self.parameter_initializer.initialize_state_populations()
        )
        # print("\nState Populations:\n", self.state_populations)

        # To help avoid rapidly changing policies.
        (
            self.action_histories,
            self.previous_actions,
            self.current_actions,
            self.allowed_actions,
            self.required_actions,
            self.allowed_actions_numbers,
            self.no_npm_pm_counters,
            self.sdm_counters,
            self.lockdown_counters,
            self.mask_mandate_counters,
            self.vaccination_mandate_counters,
        ) = self.parameter_initializer.initialize_action_dynamics(
            action_space=self.action_space
        )

        # print("\nAction Histories:\n", self.action_histories)
        # print("\nPrevious Actions:\n", self.previous_actions)
        # print("\nCurrent Actions:\n", self.current_actions)
        # print("\nAllowed Actions:\n", self.allowed_actions)
        # print("\nRequired Actions:\n", self.required_actions)
        # print("\nAllowed Action Numbers:\n", self.allowed_actions_numbers)

        # print("\nAction Counters:\n", self.no_npm_pm_counter, self.sdm_counter, self.lockdown_counter,
        #       self.mask_mandate_counter, self.vaccination_mandate_counter)

        # Hyperparameters for reward function.
        self.economic_and_social_rate_lower_limit = 70
        self.economic_and_social_rate_coefficient = 1
        self.infection_coefficient = 500_000
        self.penalty_coefficient = 1_000
        self.deceased_coefficient = 10_000

        self.max_timesteps = 181
        self.timestep = 0

        self.min_no_npm_pm_period = 14
        self.min_sdm_period = 28
        self.min_lockdown_period = 14
        self.min_mask_mandate_period = 28
        self.min_vaccination_mandate_period = 0

        self.max_no_npm_pm_period = 56
        self.max_sdm_period = 112
        self.max_lockdown_period = 42
        self.max_mask_mandate_period = 180
        self.max_vaccination_mandate_period = 0

        self.new_cases = {}
        for state in self.states:
            self.new_cases[state] = []

        self.population_dynamics_computer = PopulationDynamicsComputer()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        """This method resets the environment and returns the state as the observation.

        :returns observation: - (Vector containing the normalized count of number of healthy people, infected people
                                and hospitalized people.)"""

        self.population_dynamics_dataframes = (
            self.parameter_initializer.initialize_population_dynamics()
        )

        self.new_cases = {}
        for state in self.states:
            self.new_cases[state] = []

        self.timestep = 0

        # To help avoid rapidly changing policies.
        (
            self.action_histories,
            self.previous_actions,
            self.current_actions,
            self.allowed_actions,
            self.required_actions,
            self.allowed_actions_numbers,
            self.no_npm_pm_counters,
            self.sdm_counters,
            self.lockdown_counters,
            self.mask_mandate_counters,
            self.vaccination_mandate_counters,
        ) = self.parameter_initializer.initialize_action_dynamics(
            action_space=self.action_space
        )

        observations = {}
        for state in self.epidemiological_model_data:
            state_observation = [
                self.population_dynamics_dataframes[state]["Infected"].iloc[-1]
                / self.state_populations[state],
                self.population_dynamics_dataframes[state][
                    "Economic and Public Perception Rate"
                ].iloc[-1],
                self.previous_actions[state],
                self.current_actions[state],
            ]
            observations[state] = state_observation

        info = {}

        return observations, info

    def step(self, actions):
        """This method implements what happens when the agent takes a particular action. It changes the rate at which
        new people are infected, defines the rewards for the various states, and determines when the episode ends.

        :param actions: - Dictionary ....

        :returns observation: - (Vector containing the normalized count of number of healthy people, infected people
                                and hospitalized people.)
                 reward: - (Float value that's used to measure the performance of the agent.)
                 done: - (Boolean describing whether the episode has ended.)
                 info: - (A dictionary that can be used to provide additional implementation information.)
        """

        observations = {}
        rewards = {}
        truncations = {}
        terminations = {}
        infos = {}

        for state in self.states:
            print("State:", state)
            self.action_histories[state].append(actions[state])

            if len(self.action_histories[state]) == 1:
                self.previous_actions[state] = 0
            else:
                self.previous_actions[state] = self.action_histories[state][-2]
            self.current_actions[state] = actions[state]

            # This index helps to use the different parameter values for the different splits.
            # AD: Converts range(n) into [7 (10x), 8 (28x), 9 (28x), 10 (28x), ...]
            #     28 = 4 weeks. 214 = start date (october)
            index = int(np.floor((self.timestep + 214) / 28))

            # Updating the action dependent parameters:
            # Switch from a discrete action space to a multi-discrete action space.
            beta = None
            economic_and_public_perception_rate = None
            if actions[state] == 0:  # No NPM or PM taken. 7.3
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 1.4
                    if self.population_dynamics_dataframes[state]["Infected"].iloc[-1]
                    / self.state_populations[state]
                    >= 0.001
                    else self.epidemiological_model_parameters[state]["beta"][index]
                    * 1.1
                )

                economic_and_public_perception_rate = (
                    min(
                        1.005
                        * self.population_dynamics_dataframes[state][
                            "Economic and Public Perception Rate"
                        ].iloc[-1],
                        100,
                    )
                    if self.population_dynamics_dataframes[state]["Infected"].iloc[-1]
                    / self.state_populations[state]
                    < 0.001
                    else 0.999
                    * self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                )
                self.no_npm_pm_counters[state] += 1
                self.sdm_counters[state] = 0
                self.lockdown_counters[state] = 0
                self.mask_mandate_counters[state] = 0
                self.vaccination_mandate_counters[state] = 0

            elif actions[state] == 1:  # SDM
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.95
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.9965
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] += 1
                self.lockdown_counters[state] = 0
                self.mask_mandate_counters[state] = 0
                self.vaccination_mandate_counters[state] = 0

            elif (
                actions[state] == 2
            ):  # Lockdown (Closure of non-essential business, schools, gyms...) 0.997
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.85
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.997
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] = 0
                self.lockdown_counters[state] += 1
                self.mask_mandate_counters[state] = 0
                self.vaccination_mandate_counters[state] = 0

            elif actions[state] == 3:  # Public Mask Mandates 0.9975
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.925
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.9965
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] = 0
                self.lockdown_counters[state] = 0
                self.mask_mandate_counters[state] += 1
                self.vaccination_mandate_counters[state] = 0

            elif actions[state] == 4:  # Vaccination Mandates 0.994
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.95
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.994
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] = 0
                self.lockdown_counters[state] = 0
                self.mask_mandate_counters[state] = 0
                self.vaccination_mandate_counters[state] += 1

            elif actions[state] == 5:  # SDM and Public Mask Mandates 0.9965
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.875
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.9965
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] += 1
                self.lockdown_counters[state] = 0
                self.mask_mandate_counters[state] += 1
                self.vaccination_mandate_counters[state] = 0

            elif actions[state] == 6:  # SDM and Vaccination Mandates 0.993
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.825
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.993
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] += 1
                self.lockdown_counters[state] = 0
                self.mask_mandate_counters[state] = 0
                self.vaccination_mandate_counters[state] += 1

            elif actions[state] == 7:  # Lockdown and Public Mask Mandates 0.9965
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.75
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.994
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] = 0
                self.lockdown_counters[state] += 1
                self.mask_mandate_counters[state] += 1
                self.vaccination_mandate_counters[state] = 0

            elif actions[state] == 8:  # Lockdown and Vaccination Mandates 0.993
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.80
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.993
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] = 0
                self.lockdown_counters[state] += 1
                self.mask_mandate_counters[state] = 0
                self.vaccination_mandate_counters[state] += 1

            elif (
                actions[state] == 9
            ):  # Public Mask Mandates and Vaccination Mandates 0.9935
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.90
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.9935
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] = 0
                self.lockdown_counters[state] = 0
                self.mask_mandate_counters[state] += 1
                self.vaccination_mandate_counters[state] += 1

            elif (
                actions[state] == 10
            ):  # SDM, Public Mask Mandates and Vaccination Mandates 0.9925
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.60
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.9925
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] += 1
                self.lockdown_counters[state] = 0
                self.mask_mandate_counters[state] += 1
                self.vaccination_mandate_counters[state] += 1

            elif (
                actions[state] == 11
            ):  # Lockdown, Public Mask Mandates and Vaccination Mandates 0.9925
                beta = (
                    self.epidemiological_model_parameters[state]["beta"][index] * 0.60
                )
                economic_and_public_perception_rate = (
                    self.population_dynamics_dataframes[state][
                        "Economic and Public Perception Rate"
                    ].iloc[-1]
                    * 0.9925
                )
                self.no_npm_pm_counters[state] = 0
                self.sdm_counters[state] = 0
                self.lockdown_counters[state] += 1
                self.mask_mandate_counters[state] += 1
                self.vaccination_mandate_counters[state] += 1

            else:
                print("Invalid Action")

            (
                self.population_dynamics_dataframes,
                self.new_cases,
            ) = self.population_dynamics_computer.compute_population_dynamics(
                action=actions[state],
                beta=beta,
                environment_config=self.environment_config,
                epidemiological_model_data=self.epidemiological_model_data,
                epidemiological_model_parameters=self.epidemiological_model_parameters,
                new_cases=self.new_cases,
                population_dynamics_dataframes=self.population_dynamics_dataframes,
                state=state,
                state_populations=self.state_populations,
                timestep=self.timestep,
            )

            self.population_dynamics_dataframes[state][
                "Economic and Public Perception Rate"
            ].iloc[self.timestep + 1] = economic_and_public_perception_rate
            print("\n\nAfter EPP:\n", self.population_dynamics_dataframes[state])

            # Checking which actions are allowed:
            # Potential Violations (If the action is not taken in the next time-step.):
            no_npm_pm_min_period_violation = (
                True
                if (0 < self.no_npm_pm_counters[state] < self.min_no_npm_pm_period)
                else False
            )
            sdm_min_period_violation = (
                True if (0 < self.sdm_counters[state] < self.min_sdm_period) else False
            )
            lockdown_min_period_violation = (
                True
                if (0 < self.lockdown_counters[state] < self.min_lockdown_period)
                else False
            )
            mask_mandate_min_period_violation = (
                True
                if (
                    0 < self.mask_mandate_counters[state] < self.min_mask_mandate_period
                )
                else False
            )
            vaccination_mandate_min_period_violation = (
                True
                if (
                    0
                    < self.vaccination_mandate_counters[state]
                    < self.min_vaccination_mandate_period
                )
                else False
            )

            # Potential Violations (If the action is taken in the next time-step.):
            no_npm_pm_max_period_violation = (
                True
                if (self.no_npm_pm_counters[state] >= self.max_no_npm_pm_period)
                else False
            )
            sdm_max_period_violation = (
                True if (self.sdm_counters[state] >= self.max_sdm_period) else False
            )
            lockdown_max_period_violation = (
                True
                if (self.lockdown_counters[state] >= self.max_lockdown_period)
                else False
            )
            mask_mandate_max_period_violation = (
                True
                if (self.mask_mandate_counters[state] >= self.max_mask_mandate_period)
                else False
            )
            vaccination_mandate_max_period_violation = (
                True
                if (
                    self.vaccination_mandate_counters[state]
                    >= self.max_vaccination_mandate_period
                )
                else False
            )

            # Required Actions (As in not taking them will result in minimum violation):
            no_npm_pm_required = True if no_npm_pm_min_period_violation else False
            sdm_required = True if sdm_min_period_violation else False
            lockdown_required = True if lockdown_min_period_violation else False
            mask_mandate_required = True if mask_mandate_min_period_violation else False
            vaccination_mandate_required = (
                True if vaccination_mandate_min_period_violation else False
            )

            # Allowed Actions
            no_npm_pm_allowed = (
                True
                if (
                    (not sdm_min_period_violation)
                    and (not lockdown_min_period_violation)
                    and (not mask_mandate_min_period_violation)
                    and (not vaccination_mandate_min_period_violation)
                    and (not no_npm_pm_max_period_violation)
                )
                else False
            )

            sdm_allowed = (
                True
                if (
                    (not no_npm_pm_min_period_violation)
                    and (not lockdown_min_period_violation)
                    and (not sdm_max_period_violation)
                )
                else False
            )

            lockdown_allowed = (
                True
                if (
                    (not no_npm_pm_min_period_violation)
                    and (not sdm_min_period_violation)
                    and (not lockdown_max_period_violation)
                )
                else False
            )

            mask_mandate_allowed = (
                True
                if (
                    (not no_npm_pm_min_period_violation)
                    and (not mask_mandate_max_period_violation)
                )
                else False
            )

            vaccination_mandate_allowed = (
                True
                if (
                    (not no_npm_pm_min_period_violation)
                    and (not vaccination_mandate_max_period_violation)
                )
                else False
            )

            # Updating the lists for allowed and required actions.
            self.allowed_actions[state] = [
                no_npm_pm_allowed,
                sdm_allowed,
                lockdown_allowed,
                mask_mandate_allowed,
                vaccination_mandate_allowed,
            ]
            self.required_actions[state] = [
                no_npm_pm_required,
                sdm_required,
                lockdown_required,
                mask_mandate_required,
                vaccination_mandate_required,
            ]

            # Logic to determine which action as per the numbers is allowed.
            # (Each list within the list is a set of actions corresponding to the five "isolated" actions.)
            action_association_list = [
                [0],
                [1, 5, 6, 10],
                [2, 7, 8, 11],
                [3, 5, 7, 9, 10, 11],
                [4, 6, 8, 9, 10, 11],
            ]
            actions_allowed = None

            # First we simply go through the required actions and find the set of associated actions. This can lead to a
            # situation in which for e.g., the mask mandate action is required but not all other actions in the action
            # association list such as lockdown are allowed are included. We remove them with the next for loop.
            for i in range(5):
                if self.required_actions[state][i]:
                    if actions_allowed is None:
                        actions_allowed = set(action_association_list[i])
                    else:
                        # Set intersection operator.
                        actions_allowed = actions_allowed & set(
                            action_association_list[i]
                        )

            # Here we check if the "actions_allowed" set contains any actions that are in fact not allowed
            # (and not required). We remove such actions from the set with by taking a difference between the sets.
            for i in range(5):
                if (
                    not self.allowed_actions[state][i]
                    and not self.required_actions[state][i]
                ):
                    if actions_allowed is None:
                        break
                    else:
                        actions_allowed = actions_allowed.difference(
                            set(action_association_list[i])
                        )

            # Exception case.
            if actions_allowed is None:
                for i in range(5):
                    if self.allowed_actions[state][i]:
                        if actions_allowed is None:
                            actions_allowed = set(action_association_list[i])
                        else:
                            actions_allowed = actions_allowed.union(
                                set(action_association_list[i])
                            )
                for i in range(5):
                    if not self.allowed_actions[state][i]:
                        actions_allowed = actions_allowed.difference(
                            set(action_association_list[i])
                        )

            actions_allowed = list(actions_allowed)
            self.allowed_actions_numbers[state] = [
                1 if i in actions_allowed else 0 for i in range(self.action_space.n)
            ]

            # Reward
            rewards[state] = (
                -self.infection_coefficient
                * self.population_dynamics_dataframes[state]["Infected"].iloc[-1]
                / self.state_populations[state]
                + self.population_dynamics_dataframes[state][
                    "Economic and Public Perception Rate"
                ].iloc[-1]
            )

            state_observation = [
                self.population_dynamics_dataframes[state]["Infected"].iloc[-1]
                / self.state_populations[state],
                self.population_dynamics_dataframes[state][
                    "Economic and Public Perception Rate"
                ].iloc[-1],
                self.previous_actions[state],
                self.current_actions[state],
            ]
            observations[state] = state_observation

            # The episode terminates when the number of infected people becomes greater than 25 % of the population.
            terminations[state] = (
                True
                if (
                    self.population_dynamics_dataframes[state]["Infected"].iloc[-1]
                    >= 0.99 * self.state_populations[state]
                    or self.timestep >= self.max_timesteps
                )
                else False
            )
            truncations[state] = False
            infos[state] = {}

        print("Timestep Before:", self.timestep)
        self.timestep += 1
        print("Timestep After:", self.timestep)

        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        """This method renders the statistical graph of the population.

        :param mode: 'human' renders to the current display or terminal and returns nothing.
        """

        return


# TEST
environment_configuration = {
    "data_path": f"{DATA_DIR}/epidemiological_model_data/",
    "simulation_start_date": "11/01/2021",
}
epidemic_simulation = EpidemicSimulationMA(env_config=environment_configuration)
epidemic_simulation.step({"New York": 0, "Pennsylvania": 0})
