import pandas as pd
from src.settings import data_directory
import json
from pathlib import Path
import os
from lmfit import Parameters


class ParameterInitializer:
    def __init__(self, data_path, simulation_start_date=None):
        """This method initializes the required variables."""

        self.data_path = data_path
        self.simulation_start_date = simulation_start_date
        self.epidemiological_model_data = {}
        self.states = self.initialize_state_names()

    def initialize_state_names(self):
        """This method initializes the states names."""

        states = []
        for root, directory_names, filenames in os.walk(self.data_path):
            for filename in filenames:
                states.append(Path(filename).stem)

        return states

    def initialize_epidemiological_model_data(self):
        """This method initializes the epidemiological model data."""

        for state_name in self.states:
            self.epidemiological_model_data[state_name] = pd.read_csv(
                f"{self.data_path}/{state_name}.csv"
            )

        return self.epidemiological_model_data

    def initialize_epidemiological_model_parameters(self):
        """This method initializes the epidemiological model parameters."""

        epidemiological_model_parameters = {}
        for state in self.states:
            file = open(
                f"{data_directory}/epidemiological_model_parameters/{state}.json"
            )
            data = json.load(file)
            epidemiological_model_parameters[state] = data

        return epidemiological_model_parameters

    def initialize_population_dynamics(self):
        """This method initializes the population dynamics."""

        population_dynamics = {}

        for state in self.epidemiological_model_data:
            self.epidemiological_model_data[state]["date"] = pd.to_datetime(
                self.epidemiological_model_data[state]["date"]
            )

            # Population Dynamics by Epidemiological Compartments:
            number_of_susceptible_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Susceptible",
                ]
                .iloc[0]
            )
            number_of_exposed_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Exposed",
                ]
                .iloc[0]
            )
            number_of_infected_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Infected",
                ]
                .iloc[0]
            )
            number_of_hospitalized_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Hospitalized",
                ]
                .iloc[0]
            )
            number_of_recovered_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Recovered",
                ]
                .iloc[0]
            )
            number_of_deceased_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Deceased",
                ]
                .iloc[0]
            )

            # Population Dynamics by Vaccination Status:
            number_of_unvaccinated_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "unvaccinated_individuals",
                ]
                .iloc[0]
            )
            number_of_fully_vaccinated_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "fully_vaccinated_individuals",
                ]
                .iloc[0]
            )
            number_of_booster_vaccinated_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "boosted_individuals",
                ]
                .iloc[0]
            )

            # Susceptible Compartment by Vaccination Status:
            number_of_unvaccinated_susceptible_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Susceptible_UV",
                ]
                .iloc[0]
            )
            number_of_fully_vaccinated_susceptible_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Susceptible_FV",
                ]
                .iloc[0]
            )
            number_of_booster_vaccinated_susceptible_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Susceptible_BV",
                ]
                .iloc[0]
            )

            # Exposed Compartment by Vaccination Status:
            number_of_unvaccinated_exposed_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Exposed_UV",
                ]
                .iloc[0]
            )
            number_of_fully_vaccinated_exposed_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Exposed_FV",
                ]
                .iloc[0]
            )
            number_of_booster_vaccinated_exposed_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Exposed_BV",
                ]
                .iloc[0]
            )

            # Infected Compartment by Vaccination Status:
            number_of_unvaccinated_infected_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Infected_UV",
                ]
                .iloc[0]
            )
            number_of_fully_vaccinated_infected_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Infected_FV",
                ]
                .iloc[0]
            )
            number_of_booster_vaccinated_infected_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Infected_BV",
                ]
                .iloc[0]
            )

            # Hospitalized Compartment by Vaccination Status:
            number_of_unvaccinated_hospitalized_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Hospitalized_UV",
                ]
                .iloc[0]
            )
            number_of_fully_vaccinated_hospitalized_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Hospitalized_FV",
                ]
                .iloc[0]
            )
            number_of_booster_vaccinated_hospitalized_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Hospitalized_BV",
                ]
                .iloc[0]
            )

            # Recovered Compartment by Vaccination Status:
            number_of_unvaccinated_recovered_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Recovered_UV",
                ]
                .iloc[0]
            )
            number_of_fully_vaccinated_recovered_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Recovered_FV",
                ]
                .iloc[0]
            )
            number_of_booster_vaccinated_recovered_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Recovered_BV",
                ]
                .iloc[0]
            )

            # Deceased Compartment by Vaccination Status:
            number_of_unvaccinated_deceased_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Deceased_UV",
                ]
                .iloc[0]
            )
            number_of_fully_vaccinated_deceased_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Deceased_FV",
                ]
                .iloc[0]
            )
            number_of_booster_vaccinated_deceased_individuals = (
                self.epidemiological_model_data[state]
                .loc[
                    self.epidemiological_model_data[state]["date"]
                    == self.simulation_start_date,
                    "Deceased_BV",
                ]
                .iloc[0]
            )

            economic_and_public_perception_rate = 100.0

            state_population_dynamics = pd.DataFrame(
                data=list(
                    zip(
                        [self.simulation_start_date],
                        [number_of_unvaccinated_individuals],
                        [number_of_fully_vaccinated_individuals],
                        [number_of_booster_vaccinated_individuals],
                        [number_of_susceptible_individuals],
                        [number_of_exposed_individuals],
                        [number_of_infected_individuals],
                        [number_of_hospitalized_individuals],
                        [number_of_recovered_individuals],
                        [number_of_deceased_individuals],
                        [number_of_unvaccinated_susceptible_individuals],
                        [number_of_fully_vaccinated_susceptible_individuals],
                        [number_of_booster_vaccinated_susceptible_individuals],
                        [number_of_unvaccinated_exposed_individuals],
                        [number_of_fully_vaccinated_exposed_individuals],
                        [number_of_booster_vaccinated_exposed_individuals],
                        [number_of_unvaccinated_infected_individuals],
                        [number_of_fully_vaccinated_infected_individuals],
                        [number_of_booster_vaccinated_infected_individuals],
                        [number_of_unvaccinated_hospitalized_individuals],
                        [number_of_fully_vaccinated_hospitalized_individuals],
                        [number_of_booster_vaccinated_hospitalized_individuals],
                        [number_of_unvaccinated_recovered_individuals],
                        [number_of_fully_vaccinated_recovered_individuals],
                        [number_of_booster_vaccinated_recovered_individuals],
                        [number_of_unvaccinated_deceased_individuals],
                        [number_of_fully_vaccinated_deceased_individuals],
                        [number_of_booster_vaccinated_deceased_individuals],
                        [economic_and_public_perception_rate],
                    )
                ),
                columns=[
                    "date",
                    "unvaccinated_individuals",
                    "fully_vaccinated_individuals",
                    "booster_vaccinated_individuals",
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
                ],
            )

            population_dynamics[state] = state_population_dynamics

        return population_dynamics

    def initialize_state_populations(self):
        """This method initializes the state populations."""

        us_population = pd.read_csv(f"{data_directory}/population/us_population.csv")
        state_populations = {}
        for state in self.states:
            state_population = us_population.loc[
                us_population["Geographic Area"] == state, "7/1/2021"
            ].iloc[0]
            state_populations[state] = int(state_population.replace(",", ""))

        return state_populations

    def initialize_action_dynamics(self, action_space):
        """This method initializes the action dynamics."""

        no_npm_pm_counters = {}
        sdm_counters = {}
        lockdown_counters = {}
        mask_mandate_counters = {}
        vaccination_mandate_counters = {}
        action_histories = {}
        previous_actions = {}
        current_actions = {}
        allowed_actions = {}
        required_actions = {}
        allowed_actions_numbers = {}

        for state in self.states:
            action_histories[state] = []
            previous_actions[state] = 0
            current_actions[state] = 0

            allowed_actions[state] = [True, True, True, True, True]
            required_actions[state] = [False, False, False, False, False]
            allowed_actions_numbers[state] = [1 for _ in range(action_space.n)]

            no_npm_pm_counters[state] = 0
            sdm_counters[state] = 0
            lockdown_counters[state] = 0
            mask_mandate_counters[state] = 0
            vaccination_mandate_counters[state] = 0

        return (
            action_histories,
            previous_actions,
            current_actions,
            allowed_actions,
            required_actions,
            allowed_actions_numbers,
            no_npm_pm_counters,
            sdm_counters,
            lockdown_counters,
            mask_mandate_counters,
            vaccination_mandate_counters,
        )

    @staticmethod
    def initialize_initial_epidemiological_model_parameters(
        constrained_beta=True,
    ):
        """This method initializes the initial epidemiological model parameter values for the epidemiological model
        parameter computer.

        :return parameters: Parameters for the epidemiological model."""

        # Set parameters including bounds
        parameters = Parameters()

        # # Exposure rate.
        # if constrained_beta:
        #     parameters.add("beta", value=3, min=0.0, max=10)
        # else:
        #     parameters.add("beta", value=3, min=1.5, max=5)

        # Population mixing coefficient.
        parameters.add("alpha", value=0.85, min=0.7, max=1)

        # Infection rates.
        # parameters.add("beta_uv", value=0.02, min=0, max=0.05)
        # parameters.add("beta_v", value=0.003, min=0, max=0.007)
        # parameters.add("beta_biv", value=0.0003, min=0, max=0.007)
        parameters.add("beta_uv", value=0.02, min=0, max=10)
        parameters.add("beta_v", value=0.003, min=0, max=10)
        parameters.add("beta_biv", value=0.0003, min=0, max=10)
        parameters.add("beta_ruv", value=0.007, min=0, max=10)
        parameters.add("beta_rv", value=0.004, min=0, max=10)
        parameters.add("beta_rbiv", value=0.004, min=0, max=10)

        parameters.add("beta_5_17_uv", value=0.02, min=0, max=10)
        parameters.add("beta_5_17_v", value=0.003, min=0, max=10)
        parameters.add("beta_5_17_biv", value=0.0003, min=0, max=10)
        parameters.add("beta_5_17_ruv", value=0.007, min=0, max=10)
        parameters.add("beta_5_17_rv", value=0.004, min=0, max=10)
        parameters.add("beta_5_17_rbiv", value=0.004, min=0, max=10)

        parameters.add("beta_18_49_uv", value=0.02, min=0, max=10)
        parameters.add("beta_18_49_v", value=0.003, min=0, max=10)
        parameters.add("beta_18_49_biv", value=0.0003, min=0, max=10)
        parameters.add("beta_18_49_ruv", value=0.007, min=0, max=10)
        parameters.add("beta_18_49_rv", value=0.004, min=0, max=10)
        parameters.add("beta_18_49_rbiv", value=0.004, min=0, max=10)

        parameters.add("beta_50_64_uv", value=0.02, min=0, max=10)
        parameters.add("beta_50_64_v", value=0.003, min=0, max=10)
        parameters.add("beta_50_64_biv", value=0.0003, min=0, max=10)
        parameters.add("beta_50_64_ruv", value=0.007, min=0, max=10)
        parameters.add("beta_50_64_rv", value=0.004, min=0, max=10)
        parameters.add("beta_50_64_rbiv", value=0.004, min=0, max=10)

        parameters.add("beta_65_plus_uv", value=0.02, min=0, max=10)
        parameters.add("beta_65_plus_v", value=0.003, min=0, max=10)
        parameters.add("beta_65_plus_biv", value=0.0003, min=0, max=10)
        parameters.add("beta_65_plus_ruv", value=0.007, min=0, max=10)
        parameters.add("beta_65_plus_rv", value=0.004, min=0, max=10)
        parameters.add("beta_65_plus_rbiv", value=0.004, min=0, max=10)

        # Hospitalization rates for infected individuals.
        parameters.add("delta_uv", value=0.00216666, min=0.0, max=0.00444444)
        parameters.add("delta_v", value=0.000516666, min=0.0, max=0.00444444)
        parameters.add("delta_biv", value=0.000516666, min=0.0, max=0.00444444)

        parameters.add("delta_5_17_uv", value=0.00216666, min=0.0, max=0.00444444)
        parameters.add("delta_5_17_v", value=0.000516666, min=0.0, max=0.00444444)
        parameters.add("delta_5_17_biv", value=0.000516666, min=0.0, max=0.00444444)

        parameters.add("delta_18_49_uv", value=0.00216666, min=0.0, max=0.00444444)
        parameters.add("delta_18_49_v", value=0.000516666, min=0.0, max=0.00444444)
        parameters.add("delta_18_49_biv", value=0.000516666, min=0.0, max=0.00444444)

        parameters.add("delta_50_64_uv", value=0.00216666, min=0.0, max=0.00444444)
        parameters.add("delta_50_64_v", value=0.000516666, min=0.0, max=0.00444444)
        parameters.add("delta_50_64_biv", value=0.000516666, min=0.0, max=0.00444444)

        parameters.add("delta_65_plus_uv", value=0.00216666, min=0.0, max=0.00444444)
        parameters.add("delta_65_plus_v", value=0.000516666, min=0.0, max=0.00444444)
        parameters.add("delta_65_plus_biv", value=0.000516666, min=0.0, max=0.00444444)

        # Recovery rates for infected individuals.
        parameters.add("gamma_i_uv", value=0.05, min=0.040, max=0.055)
        parameters.add("gamma_i_v", value=0.053, min=0.045, max=0.055)
        parameters.add("gamma_i_biv", value=0.053, min=0.0475, max=0.065)
        parameters.add("gamma_h_uv", value=0.0277777, min=0.025, max=0.055)
        parameters.add("gamma_h_v", value=0.0377777, min=0.03, max=0.055)
        parameters.add("gamma_h_biv", value=0.0377777, min=0.03, max=0.065)

        parameters.add("gamma_i_5_17_uv", value=0.05, min=0.040, max=0.055)
        parameters.add("gamma_i_5_17_v", value=0.053, min=0.045, max=0.055)
        parameters.add("gamma_i_5_17_biv", value=0.053, min=0.0475, max=0.065)
        parameters.add("gamma_h_5_17_uv", value=0.0277777, min=0.025, max=0.055)
        parameters.add("gamma_h_5_17_v", value=0.0377777, min=0.03, max=0.055)
        parameters.add("gamma_h_5_17_biv", value=0.0377777, min=0.03, max=0.065)

        parameters.add("gamma_i_18_49_uv", value=0.05, min=0.040, max=0.055)
        parameters.add("gamma_i_18_49_v", value=0.053, min=0.045, max=0.055)
        parameters.add("gamma_i_18_49_biv", value=0.053, min=0.0475, max=0.065)
        parameters.add("gamma_h_18_49_uv", value=0.0277777, min=0.025, max=0.055)
        parameters.add("gamma_h_18_49_v", value=0.0377777, min=0.03, max=0.055)
        parameters.add("gamma_h_18_49_biv", value=0.0377777, min=0.03, max=0.065)

        parameters.add("gamma_i_50_64_uv", value=0.05, min=0.040, max=0.055)
        parameters.add("gamma_i_50_64_v", value=0.053, min=0.045, max=0.055)
        parameters.add("gamma_i_50_64_biv", value=0.053, min=0.0475, max=0.065)
        parameters.add("gamma_h_50_64_uv", value=0.0277777, min=0.025, max=0.055)
        parameters.add("gamma_h_50_64_v", value=0.0377777, min=0.03, max=0.055)
        parameters.add("gamma_h_50_64_biv", value=0.0377777, min=0.03, max=0.065)

        parameters.add("gamma_i_65_plus_uv", value=0.05, min=0.040, max=0.055)
        parameters.add("gamma_i_65_plus_v", value=0.053, min=0.045, max=0.055)
        parameters.add("gamma_i_65_plus_biv", value=0.053, min=0.0475, max=0.065)
        parameters.add("gamma_h_65_plus_uv", value=0.0277777, min=0.025, max=0.055)
        parameters.add("gamma_h_65_plus_v", value=0.0377777, min=0.03, max=0.055)
        parameters.add("gamma_h_65_plus_biv", value=0.0377777, min=0.03, max=0.065)

        # Death rates for infected individuals.
        parameters.add(
            "mu_i_uv", value=0.00155555555, min=0.00005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_v", value=0.000005555555, min=0.000005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_biv", value=0.000005555555, min=0.000002555555, max=0.0033333333
        )
        parameters.add("mu_h_uv", value=0.00877777, min=0.00277777, max=0.01388888)
        parameters.add("mu_h_v", value=0.00277777, min=0.00077777, max=0.01388888)
        parameters.add("mu_h_biv", value=0.00087777, min=0.000177777, max=0.01388888)

        parameters.add(
            "mu_i_5_17_uv", value=0.00155555555, min=0.00005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_5_17_v", value=0.000005555555, min=0.000005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_5_17_biv", value=0.000005555555, min=0.000002555555, max=0.0033333333
        )
        parameters.add("mu_h_5_17_uv", value=0.00877777, min=0.00277777, max=0.01388888)
        parameters.add("mu_h_5_17_v", value=0.00277777, min=0.00077777, max=0.01388888)
        parameters.add(
            "mu_h_5_17_biv", value=0.00087777, min=0.000177777, max=0.01388888
        )

        parameters.add(
            "mu_i_18_49_uv", value=0.00155555555, min=0.00005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_18_49_v", value=0.000005555555, min=0.000005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_18_49_biv", value=0.000005555555, min=0.000002555555, max=0.0033333333
        )
        parameters.add(
            "mu_h_18_49_uv", value=0.00877777, min=0.00277777, max=0.01388888
        )
        parameters.add("mu_h_18_49_v", value=0.00277777, min=0.00077777, max=0.01388888)
        parameters.add(
            "mu_h_18_49_biv", value=0.00087777, min=0.000177777, max=0.01388888
        )

        parameters.add(
            "mu_i_50_64_uv", value=0.00155555555, min=0.00005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_50_64_v", value=0.000005555555, min=0.000005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_50_64_biv", value=0.000005555555, min=0.000002555555, max=0.0033333333
        )
        parameters.add(
            "mu_h_50_64_uv", value=0.00877777, min=0.00277777, max=0.01388888
        )
        parameters.add("mu_h_50_64_v", value=0.00277777, min=0.00077777, max=0.01388888)
        parameters.add(
            "mu_h_50_64_biv", value=0.00087777, min=0.000177777, max=0.01388888
        )

        parameters.add(
            "mu_i_65_plus_uv", value=0.00155555555, min=0.00005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_65_plus_v", value=0.000005555555, min=0.000005555555, max=0.0033333333
        )
        parameters.add(
            "mu_i_65_plus_biv",
            value=0.000005555555,
            min=0.000002555555,
            max=0.0033333333,
        )
        parameters.add(
            "mu_h_65_plus_uv", value=0.00877777, min=0.00277777, max=0.01388888
        )
        parameters.add(
            "mu_h_65_plus_v", value=0.00277777, min=0.00077777, max=0.01388888
        )
        parameters.add(
            "mu_h_65_plus_biv", value=0.00087777, min=0.000177777, max=0.01388888
        )

        return parameters
