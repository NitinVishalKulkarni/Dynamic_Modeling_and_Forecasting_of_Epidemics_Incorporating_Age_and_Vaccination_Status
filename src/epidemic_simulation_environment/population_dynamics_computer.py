import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epftoolbox.evaluation import sMAPE
from pandas.tseries.offsets import DateOffset
from scipy.stats import hmean
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from multiprocessing import Pool
from src.settings import data_directory
from src.utilities.parameter_initializer import ParameterInitializer


class PopulationDynamicsComputer:
    def __init__(self, population_dynamics_computer_configuration):
        self.population_dynamics_computer_configuration = (
            population_dynamics_computer_configuration
        )

        self.parameter_initializer = ParameterInitializer(
            data_path=f"{data_directory}/epidemiological_model_data/"
        )

        self.states = self.parameter_initializer.initialize_state_names()

        self.epidemiological_model_data = (
            self.parameter_initializer.initialize_epidemiological_model_data()
        )
        self.state_populations = (
            self.parameter_initializer.initialize_state_populations()
        )

        self.epidemiological_model_parameters = (
            self.parameter_initializer.initialize_epidemiological_model_parameters()
        )
        self.epidemiological_compartment_names = (
            population_dynamics_computer_configuration[
                "epidemiological_compartment_names"
            ]
        )

        self.simulation_data = {}
        for state_name in self.states:
            self.simulation_data[state_name] = pd.DataFrame(
                columns=self.epidemiological_compartment_names
            )

        self.average_mape = []
        self.average_smape = []
        self.average_rmse = []

    @staticmethod
    def compute_population_dynamics(
            action,
            beta,
            environment_config,
            epidemiological_model_data,
            epidemiological_model_parameters,
            new_cases,
            population_dynamics_dataframes,
            state,
            state_populations,
            timestep,
    ):
        """This method computes the action dependent population dynamics
        :parameter action: Integer - Represents the action taken by the agent
        :param beta:
        :param environment_config:
        :param epidemiological_model_data
        :param epidemiological_model_parameters
        :param new_cases
        :param population_dynamics_dataframes
        :param state
        :param state_populations
        :param timestep
        """

        # Action dependent vaccination rates.
        if action in [3, 5, 6, 7]:
            percentage_unvaccinated_to_fully_vaccinated = 0.007084760245099044
            # percentage_fully_vaccinated_to_booster_vaccinated = 0.0017285714029114
            percentage_fully_vaccinated_to_booster_vaccinated = (
                epidemiological_model_data[state][
                    "percentage_fully_vaccinated_to_boosted"
                ].iloc[timestep + 214]
            )
        else:
            percentage_unvaccinated_to_fully_vaccinated = epidemiological_model_data[
                state
            ]["percentage_unvaccinated_to_fully_vaccinated"].iloc[timestep + 214]
            percentage_fully_vaccinated_to_booster_vaccinated = (
                epidemiological_model_data[state][
                    "percentage_fully_vaccinated_to_boosted"
                ].iloc[timestep + 214]
            )

        # Index to use the different model parameter values for the different splits.
        index = int(np.floor((timestep + 214) / 28))

        standard_deviation = 0.05

        mu, sigma = beta, standard_deviation * beta
        beta = np.random.normal(mu, sigma, 1)

        for model_parameter in epidemiological_model_parameters[state]:
            if model_parameter == "beta":
                continue

            mu, sigma = (
                epidemiological_model_parameters[state][model_parameter][index],
                standard_deviation
                * epidemiological_model_parameters[state][model_parameter][index],
            )
            model_parameter_value = np.random.normal(mu, sigma, 1)[0]
            epidemiological_model_parameters[state][model_parameter][
                index
            ] = model_parameter_value

        # Susceptible Compartment
        number_of_unvaccinated_susceptible_individuals = int(
            population_dynamics_dataframes[state]["Susceptible_UV"].iloc[-1]
            - (
                    beta
                    * population_dynamics_dataframes[state]["Susceptible_UV"].iloc[-1]
                    * (
                            population_dynamics_dataframes[state]["Infected"].iloc[-1]
                            ** epidemiological_model_parameters[state]["alpha"][index]
                    )
                    / state_populations[state]
            )
            + epidemiological_model_parameters[state]["sigma_s_uv"][index]
            * population_dynamics_dataframes[state]["Susceptible_UV"].iloc[-1]
            - percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state]["Susceptible_UV"].iloc[-1]
        )

        number_of_fully_vaccinated_susceptible_individuals = int(
            population_dynamics_dataframes[state]["Susceptible_FV"].iloc[-1]
            - beta
            * population_dynamics_dataframes[state]["Susceptible_FV"].iloc[-1]
            * (
                    population_dynamics_dataframes[state]["Infected"].iloc[-1]
                    ** epidemiological_model_parameters[state]["alpha"][index]
            )
            / state_populations[state]
            + epidemiological_model_parameters[state]["sigma_s_fv"][index]
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
            + percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state]["Susceptible_UV"].iloc[-1]
            - percentage_fully_vaccinated_to_booster_vaccinated
            * population_dynamics_dataframes[state]["Susceptible_FV"].iloc[-1]
        )

        number_of_booster_vaccinated_susceptible_individuals = int(
            population_dynamics_dataframes[state]["Susceptible_BV"].iloc[-1]
            - beta
            * population_dynamics_dataframes[state]["Susceptible_BV"].iloc[-1]
            * (
                    population_dynamics_dataframes[state]["Infected"].iloc[-1]
                    ** epidemiological_model_parameters[state]["alpha"][index]
            )
            / state_populations[state]
            + epidemiological_model_parameters[state]["sigma_s_bv"][index]
            * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            + percentage_fully_vaccinated_to_booster_vaccinated
            * population_dynamics_dataframes[state]["Susceptible_FV"].iloc[-1]
        )

        number_of_susceptible_individuals = (
                number_of_unvaccinated_susceptible_individuals
                + number_of_fully_vaccinated_susceptible_individuals
                + number_of_booster_vaccinated_susceptible_individuals
        )

        # Exposed Compartment
        number_of_unvaccinated_exposed_individuals = int(
            population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
            + beta
            * population_dynamics_dataframes[state]["Susceptible_UV"].iloc[-1]
            * (
                    population_dynamics_dataframes[state]["Infected"].iloc[-1]
                    ** epidemiological_model_parameters[state]["alpha"][index]
            )
            / state_populations[state]
            + (
                    beta
                    * population_dynamics_dataframes[state]["Recovered_UV"].iloc[-1]
                    * (
                            population_dynamics_dataframes[state]["Infected"].iloc[-1]
                            ** epidemiological_model_parameters[state]["alpha"][index]
                    )
                    / state_populations[state]
            )
            - epidemiological_model_parameters[state]["zeta_s_uv"][index]
            * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
            - epidemiological_model_parameters[state]["zeta_r_uv"][index]
            * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
            - epidemiological_model_parameters[state]["sigma_s_uv"][index]
            * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
            - epidemiological_model_parameters[state]["sigma_r_uv"][index]
            * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
            - percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
        )

        number_of_fully_vaccinated_exposed_individuals = int(
            population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
            + beta
            * population_dynamics_dataframes[state]["Susceptible_FV"].iloc[-1]
            * (
                    population_dynamics_dataframes[state]["Infected"].iloc[-1]
                    ** epidemiological_model_parameters[state]["alpha"][index]
            )
            / state_populations[state]
            + (
                    beta
                    * population_dynamics_dataframes[state]["Recovered_FV"].iloc[-1]
                    * (
                            population_dynamics_dataframes[state]["Infected"].iloc[-1]
                            ** epidemiological_model_parameters[state]["alpha"][index]
                    )
                    / state_populations[state]
            )
            - epidemiological_model_parameters[state]["zeta_s_fv"][index]
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
            - epidemiological_model_parameters[state]["zeta_r_fv"][index]
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
            - epidemiological_model_parameters[state]["sigma_s_fv"][index]
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
            - epidemiological_model_parameters[state]["sigma_r_fv"][index]
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
            + percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
            - percentage_fully_vaccinated_to_booster_vaccinated
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
        )

        number_of_booster_vaccinated_exposed_individuals = int(
            population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            + beta
            * population_dynamics_dataframes[state]["Susceptible_BV"].iloc[-1]
            * (
                    population_dynamics_dataframes[state]["Infected"].iloc[-1]
                    ** epidemiological_model_parameters[state]["alpha"][index]
            )
            / state_populations[state]
            + (
                    beta
                    * population_dynamics_dataframes[state]["Recovered_BV"].iloc[-1]
                    * (
                            population_dynamics_dataframes[state]["Infected"].iloc[-1]
                            ** epidemiological_model_parameters[state]["alpha"][index]
                    )
                    / state_populations[state]
            )
            - epidemiological_model_parameters[state]["zeta_s_bv"][index]
            * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            - epidemiological_model_parameters[state]["zeta_r_bv"][index]
            * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            - epidemiological_model_parameters[state]["sigma_s_bv"][index]
            * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            - epidemiological_model_parameters[state]["sigma_r_bv"][index]
            * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            + percentage_fully_vaccinated_to_booster_vaccinated
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
        )

        number_of_exposed_individuals = (
                number_of_unvaccinated_exposed_individuals
                + number_of_fully_vaccinated_exposed_individuals
                + number_of_booster_vaccinated_exposed_individuals
        )

        # Infected Compartment
        number_of_unvaccinated_infected_individuals = int(
            population_dynamics_dataframes[state]["Infected_UV"].iloc[-1]
            + epidemiological_model_parameters[state]["zeta_s_uv"][index]
            * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
            + epidemiological_model_parameters[state]["zeta_r_uv"][index]
            * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
            - epidemiological_model_parameters[state]["delta_uv"][index]
            * population_dynamics_dataframes[state]["Infected_UV"].iloc[-1]
            - epidemiological_model_parameters[state]["gamma_i_uv"][index]
            * population_dynamics_dataframes[state]["Infected_UV"].iloc[-1]
            - epidemiological_model_parameters[state]["mu_i_uv"][index]
            * population_dynamics_dataframes[state]["Infected_UV"].iloc[-1]
        )

        number_of_fully_vaccinated_infected_individuals = int(
            population_dynamics_dataframes[state]["Infected_FV"].iloc[-1]
            + epidemiological_model_parameters[state]["zeta_s_fv"][index]
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
            + epidemiological_model_parameters[state]["zeta_r_fv"][index]
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
            - epidemiological_model_parameters[state]["delta_fv"][index]
            * population_dynamics_dataframes[state]["Infected_FV"].iloc[-1]
            - epidemiological_model_parameters[state]["gamma_i_fv"][index]
            * population_dynamics_dataframes[state]["Infected_FV"].iloc[-1]
            - epidemiological_model_parameters[state]["mu_i_fv"][index]
            * population_dynamics_dataframes[state]["Infected_FV"].iloc[-1]
        )

        number_of_booster_vaccinated_infected_individuals = int(
            population_dynamics_dataframes[state]["Infected_BV"].iloc[-1]
            + epidemiological_model_parameters[state]["zeta_s_bv"][index]
            * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            + epidemiological_model_parameters[state]["zeta_r_bv"][index]
            * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            - epidemiological_model_parameters[state]["delta_bv"][index]
            * population_dynamics_dataframes[state]["Infected_BV"].iloc[-1]
            - epidemiological_model_parameters[state]["gamma_i_bv"][index]
            * population_dynamics_dataframes[state]["Infected_BV"].iloc[-1]
            - epidemiological_model_parameters[state]["mu_i_bv"][index]
            * population_dynamics_dataframes[state]["Infected_BV"].iloc[-1]
        )

        number_of_infected_individuals = (
                number_of_unvaccinated_infected_individuals
                + number_of_fully_vaccinated_infected_individuals
                + number_of_booster_vaccinated_infected_individuals
        )

        # DOUBLE CHECK:
        new_cases[state].append(
            int(
                epidemiological_model_parameters[state]["zeta_s_uv"][index]
                * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
                + epidemiological_model_parameters[state]["zeta_r_uv"][index]
                * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
                + epidemiological_model_parameters[state]["zeta_s_fv"][index]
                * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
                + epidemiological_model_parameters[state]["zeta_r_fv"][index]
                * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
                + epidemiological_model_parameters[state]["zeta_s_bv"][index]
                * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
                + epidemiological_model_parameters[state]["zeta_r_bv"][index]
                * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            )
        )

        # Hospitalized Compartment
        number_of_unvaccinated_hospitalized_individuals = int(
            population_dynamics_dataframes[state]["Hospitalized_UV"].iloc[-1]
            + epidemiological_model_parameters[state]["delta_uv"][index]
            * population_dynamics_dataframes[state]["Infected_UV"].iloc[-1]
            - epidemiological_model_parameters[state]["gamma_h_uv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_UV"].iloc[-1]
            - epidemiological_model_parameters[state]["mu_h_uv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_UV"].iloc[-1]
        )

        number_of_fully_vaccinated_hospitalized_individuals = int(
            population_dynamics_dataframes[state]["Hospitalized_FV"].iloc[-1]
            + epidemiological_model_parameters[state]["delta_fv"][index]
            * population_dynamics_dataframes[state]["Infected_FV"].iloc[-1]
            - epidemiological_model_parameters[state]["gamma_h_fv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_FV"].iloc[-1]
            - epidemiological_model_parameters[state]["mu_h_fv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_FV"].iloc[-1]
        )

        number_of_booster_vaccinated_hospitalized_individuals = int(
            population_dynamics_dataframes[state]["Hospitalized_BV"].iloc[-1]
            + epidemiological_model_parameters[state]["delta_bv"][index]
            * population_dynamics_dataframes[state]["Infected_BV"].iloc[-1]
            - epidemiological_model_parameters[state]["gamma_h_bv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_BV"].iloc[-1]
            - epidemiological_model_parameters[state]["mu_h_bv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_BV"].iloc[-1]
        )

        number_of_hospitalized_individuals = (
                number_of_unvaccinated_hospitalized_individuals
                + number_of_fully_vaccinated_hospitalized_individuals
                + number_of_booster_vaccinated_hospitalized_individuals
        )

        # Recovered Compartment
        number_of_unvaccinated_recovered_individuals = int(
            population_dynamics_dataframes[state]["Recovered_UV"].iloc[-1]
            - (
                    beta
                    * population_dynamics_dataframes[state]["Recovered_UV"].iloc[-1]
                    * (
                            population_dynamics_dataframes[state]["Infected"].iloc[-1]
                            ** epidemiological_model_parameters[state]["alpha"][index]
                    )
                    / state_populations[state]
            )
            + epidemiological_model_parameters[state]["sigma_r_uv"][index]
            * population_dynamics_dataframes[state]["Exposed_UV"].iloc[-1]
            + epidemiological_model_parameters[state]["gamma_i_uv"][index]
            * population_dynamics_dataframes[state]["Infected_UV"].iloc[-1]
            + epidemiological_model_parameters[state]["gamma_h_uv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_UV"].iloc[-1]
            - percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state]["Recovered_UV"].iloc[-1]
        )

        number_of_fully_vaccinated_recovered_individuals = int(
            population_dynamics_dataframes[state]["Recovered_FV"].iloc[-1]
            - (
                    beta
                    * population_dynamics_dataframes[state]["Recovered_FV"].iloc[-1]
                    * (
                            population_dynamics_dataframes[state]["Infected"].iloc[-1]
                            ** epidemiological_model_parameters[state]["alpha"][index]
                    )
                    / state_populations[state]
            )
            + epidemiological_model_parameters[state]["sigma_r_fv"][index]
            * population_dynamics_dataframes[state]["Exposed_FV"].iloc[-1]
            + epidemiological_model_parameters[state]["gamma_i_fv"][index]
            * population_dynamics_dataframes[state]["Infected_FV"].iloc[-1]
            + epidemiological_model_parameters[state]["gamma_h_fv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_FV"].iloc[-1]
            + percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state]["Recovered_UV"].iloc[-1]
            - percentage_fully_vaccinated_to_booster_vaccinated
            * population_dynamics_dataframes[state]["Recovered_FV"].iloc[-1]
        )

        number_of_booster_vaccinated_recovered_individuals = int(
            population_dynamics_dataframes[state]["Recovered_BV"].iloc[-1]
            - (
                    beta
                    * population_dynamics_dataframes[state]["Recovered_BV"].iloc[-1]
                    * (
                            population_dynamics_dataframes[state]["Infected"].iloc[-1]
                            ** epidemiological_model_parameters[state]["alpha"][index]
                    )
                    / state_populations[state]
            )
            + epidemiological_model_parameters[state]["sigma_r_bv"][index]
            * population_dynamics_dataframes[state]["Exposed_BV"].iloc[-1]
            + epidemiological_model_parameters[state]["gamma_i_bv"][index]
            * population_dynamics_dataframes[state]["Infected_BV"].iloc[-1]
            + epidemiological_model_parameters[state]["gamma_h_bv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_BV"].iloc[-1]
            + percentage_fully_vaccinated_to_booster_vaccinated
            * population_dynamics_dataframes[state]["Recovered_FV"].iloc[-1]
        )

        number_of_recovered_individuals = (
                number_of_unvaccinated_recovered_individuals
                + number_of_fully_vaccinated_recovered_individuals
                + number_of_booster_vaccinated_recovered_individuals
        )

        # Deceased Compartment
        number_of_unvaccinated_deceased_individuals = int(
            population_dynamics_dataframes[state]["Deceased_UV"].iloc[-1]
            + epidemiological_model_parameters[state]["mu_i_uv"][index]
            * population_dynamics_dataframes[state]["Infected_UV"].iloc[-1]
            + epidemiological_model_parameters[state]["mu_h_uv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_UV"].iloc[-1]
        )

        number_of_fully_vaccinated_deceased_individuals = int(
            population_dynamics_dataframes[state]["Deceased_FV"].iloc[-1]
            + epidemiological_model_parameters[state]["mu_i_fv"][index]
            * population_dynamics_dataframes[state]["Infected_FV"].iloc[-1]
            + epidemiological_model_parameters[state]["mu_h_fv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_FV"].iloc[-1]
        )

        number_of_booster_vaccinated_deceased_individuals = int(
            population_dynamics_dataframes[state]["Deceased_BV"].iloc[-1]
            + epidemiological_model_parameters[state]["mu_i_bv"][index]
            * population_dynamics_dataframes[state]["Infected_BV"].iloc[-1]
            + epidemiological_model_parameters[state]["mu_h_bv"][index]
            * population_dynamics_dataframes[state]["Hospitalized_BV"].iloc[-1]
        )

        number_of_deceased_individuals = (
                number_of_unvaccinated_deceased_individuals
                + number_of_fully_vaccinated_deceased_individuals
                + number_of_booster_vaccinated_deceased_individuals
        )

        # Population Dynamics by Vaccination Status
        number_of_unvaccinated_individuals = int(
            population_dynamics_dataframes[state]["unvaccinated_individuals"].iloc[-1]
            - percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state]["unvaccinated_individuals"].iloc[-1]
        )

        number_of_fully_vaccinated_individuals = int(
            population_dynamics_dataframes[state]["fully_vaccinated_individuals"].iloc[
                -1
            ]
            + percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state]["unvaccinated_individuals"].iloc[-1]
            - percentage_fully_vaccinated_to_booster_vaccinated
            * population_dynamics_dataframes[state][
                "fully_vaccinated_individuals"
            ].iloc[-1]
        )

        number_of_booster_vaccinated_individuals = int(
            population_dynamics_dataframes[state][
                "booster_vaccinated_individuals"
            ].iloc[-1]
            + percentage_fully_vaccinated_to_booster_vaccinated
            * population_dynamics_dataframes[state][
                "fully_vaccinated_individuals"
            ].iloc[-1]
        )

        # Update
        print(
            "Before:\n",
            population_dynamics_dataframes[state],
        )
        new_row = {
            "date": [environment_config["simulation_start_date"]],
            "Susceptible_UV": [number_of_unvaccinated_susceptible_individuals],
            "Susceptible_FV": [number_of_fully_vaccinated_susceptible_individuals],
            "Susceptible_BV": [number_of_booster_vaccinated_susceptible_individuals],
            "Susceptible": [number_of_susceptible_individuals],
            "Exposed_UV": [number_of_unvaccinated_exposed_individuals],
            "Exposed_FV": [number_of_fully_vaccinated_exposed_individuals],
            "Exposed_BV": [number_of_booster_vaccinated_exposed_individuals],
            "Exposed": [number_of_exposed_individuals],
            "Infected_UV": [number_of_unvaccinated_infected_individuals],
            "Infected_FV": [number_of_fully_vaccinated_infected_individuals],
            "Infected_BV": [number_of_booster_vaccinated_infected_individuals],
            "Infected": [number_of_infected_individuals],
            "Hospitalized_UV": [number_of_unvaccinated_hospitalized_individuals],
            "Hospitalized_FV": [number_of_fully_vaccinated_hospitalized_individuals],
            "Hospitalized_BV": [number_of_booster_vaccinated_hospitalized_individuals],
            "Hospitalized": [number_of_hospitalized_individuals],
            "Recovered_UV": [number_of_unvaccinated_recovered_individuals],
            "Recovered_FV": [number_of_fully_vaccinated_recovered_individuals],
            "Recovered_BV": [number_of_booster_vaccinated_recovered_individuals],
            "Recovered": [number_of_recovered_individuals],
            "Deceased_UV": [number_of_unvaccinated_deceased_individuals],
            "Deceased_FV": [number_of_fully_vaccinated_deceased_individuals],
            "Deceased_BV": [number_of_booster_vaccinated_deceased_individuals],
            "Deceased": [number_of_deceased_individuals],
            "unvaccinated_individuals": [number_of_unvaccinated_individuals],
            "fully_vaccinated_individuals": [number_of_fully_vaccinated_individuals],
            "booster_vaccinated_individuals": [
                number_of_booster_vaccinated_individuals
            ],
        }
        population_dynamics_dataframes[state] = pd.concat(
            [population_dynamics_dataframes[state], pd.DataFrame(new_row)],
            ignore_index=True,
        )
        print(
            "\n\nAfter:\n",
            population_dynamics_dataframes[state],
        )

        return (
            population_dynamics_dataframes,
            new_cases,
        )

    def epidemic_forecasting(self, state):
        """This method forecasts how an epidemic will evolve."""
        # Getting the initial values for the epidemiological model compartments.

        y0 = [
            self.epidemiological_model_data[state]
            .loc[
                pd.to_datetime(self.epidemiological_model_data[state]["date"])
                == pd.to_datetime(
                    self.population_dynamics_computer_configuration[
                        "simulation_start_date"
                    ]
                ),
                f"{self.epidemiological_compartment_names[i]}",
            ]
            .values[0]
            for i in range(len(self.epidemiological_compartment_names))
        ]

        (
            s_uv,
            s_v,
            s_biv,
            i_uv,
            i_v,
            i_biv,
            h_uv,
            h_v,
            h_biv,
            r_uv,
            r_v,
            r_biv,
            dec_uv,
            dec_v,
            dec_biv,
            s_5_17_uv,
            s_5_17_v,
            s_5_17_biv,
            s_18_49_uv,
            s_18_49_v,
            s_18_49_biv,
            s_50_64_uv,
            s_50_64_v,
            s_50_64_biv,
            s_65_plus_uv,
            s_65_plus_v,
            s_65_plus_biv,
            i_5_17_uv,
            i_5_17_v,
            i_5_17_biv,
            i_18_49_uv,
            i_18_49_v,
            i_18_49_biv,
            i_50_64_uv,
            i_50_64_v,
            i_50_64_biv,
            i_65_plus_uv,
            i_65_plus_v,
            i_65_plus_biv,
            h_5_17_uv,
            h_5_17_v,
            h_5_17_biv,
            h_18_49_uv,
            h_18_49_v,
            h_18_49_biv,
            h_50_64_uv,
            h_50_64_v,
            h_50_64_biv,
            h_65_plus_uv,
            h_65_plus_v,
            h_65_plus_biv,
            r_5_17_uv,
            r_5_17_v,
            r_5_17_biv,
            r_18_49_uv,
            r_18_49_v,
            r_18_49_biv,
            r_50_64_uv,
            r_50_64_v,
            r_50_64_biv,
            r_65_plus_uv,
            r_65_plus_v,
            r_65_plus_biv,
            dec_5_17_uv,
            dec_5_17_v,
            dec_5_17_biv,
            dec_18_49_uv,
            dec_18_49_v,
            dec_18_49_biv,
            dec_50_64_uv,
            dec_50_64_v,
            dec_50_64_biv,
            dec_65_plus_uv,
            dec_65_plus_v,
            dec_65_plus_biv,
        ) = y0

        # print(y0)
        # print(i_uv)
        # sys.exit()

        population = self.state_populations[state]

        simulation_data = self.epidemiological_model_data[state].loc[
            pd.to_datetime(self.epidemiological_model_data[state]["date"])
            >= pd.to_datetime(
                self.population_dynamics_computer_configuration["simulation_start_date"]
            )
            ]
        # print(simulation_data)
        # print("oglen",len(self.epidemiological_model_data[state]))
        # print("simlen",len(simulation_data))
        # sys.exit()

        for timestep in range(len(simulation_data)):

            if timestep % 400 == 0:
                y0 = [
                    self.epidemiological_model_data[state]
                    .loc[
                        pd.to_datetime(self.epidemiological_model_data[state]["date"])
                        == pd.to_datetime(
                            self.population_dynamics_computer_configuration[
                                "simulation_start_date"
                            ]
                        ) + DateOffset(days=timestep),
                        f"{self.epidemiological_compartment_names[i]}",
                    ]
                    .values[0]
                    for i in range(len(self.epidemiological_compartment_names))
                ]

                (
                    s_uv,
                    s_v,
                    s_biv,
                    i_uv,
                    i_v,
                    i_biv,
                    h_uv,
                    h_v,
                    h_biv,
                    r_uv,
                    r_v,
                    r_biv,
                    dec_uv,
                    dec_v,
                    dec_biv,
                    s_5_17_uv,
                    s_5_17_v,
                    s_5_17_biv,
                    s_18_49_uv,
                    s_18_49_v,
                    s_18_49_biv,
                    s_50_64_uv,
                    s_50_64_v,
                    s_50_64_biv,
                    s_65_plus_uv,
                    s_65_plus_v,
                    s_65_plus_biv,
                    i_5_17_uv,
                    i_5_17_v,
                    i_5_17_biv,
                    i_18_49_uv,
                    i_18_49_v,
                    i_18_49_biv,
                    i_50_64_uv,
                    i_50_64_v,
                    i_50_64_biv,
                    i_65_plus_uv,
                    i_65_plus_v,
                    i_65_plus_biv,
                    h_5_17_uv,
                    h_5_17_v,
                    h_5_17_biv,
                    h_18_49_uv,
                    h_18_49_v,
                    h_18_49_biv,
                    h_50_64_uv,
                    h_50_64_v,
                    h_50_64_biv,
                    h_65_plus_uv,
                    h_65_plus_v,
                    h_65_plus_biv,
                    r_5_17_uv,
                    r_5_17_v,
                    r_5_17_biv,
                    r_18_49_uv,
                    r_18_49_v,
                    r_18_49_biv,
                    r_50_64_uv,
                    r_50_64_v,
                    r_50_64_biv,
                    r_65_plus_uv,
                    r_65_plus_v,
                    r_65_plus_biv,
                    dec_5_17_uv,
                    dec_5_17_v,
                    dec_5_17_biv,
                    dec_18_49_uv,
                    dec_18_49_v,
                    dec_18_49_biv,
                    dec_50_64_uv,
                    dec_50_64_v,
                    dec_50_64_biv,
                    dec_65_plus_uv,
                    dec_65_plus_v,
                    dec_65_plus_biv,
                ) = y0

            index_param_previous_year = int(
                np.floor(
                    (
                            timestep
                            + len(self.epidemiological_model_data[state])
                            - len(simulation_data)
                            - 365
                    )
                    / 28
                )
            )

            index_param_last_28 = int(
                np.floor(
                    (len(self.epidemiological_model_data[state]) - len(simulation_data))
                    / 28
                )
                - 1
            )

            index_param = int(
                np.floor(
                    (
                            timestep
                            + len(self.epidemiological_model_data[state])
                            - len(simulation_data)
                    )
                    / 28
                )
            )

            # Loading in the computed parameters:
            parameter_values = list(
                self.epidemiological_model_parameters[state].values()
            )
            (
                alpha,
                beta_uv,
                beta_v,
                beta_biv,
                beta_ruv,
                beta_rv,
                beta_rbiv,
                beta_5_17_uv,
                beta_5_17_v,
                beta_5_17_biv,
                beta_5_17_ruv,
                beta_5_17_rv,
                beta_5_17_rbiv,
                beta_18_49_uv,
                beta_18_49_v,
                beta_18_49_biv,
                beta_18_49_ruv,
                beta_18_49_rv,
                beta_18_49_rbiv,
                beta_50_64_uv,
                beta_50_64_v,
                beta_50_64_biv,
                beta_50_64_ruv,
                beta_50_64_rv,
                beta_50_64_rbiv,
                beta_65_plus_uv,
                beta_65_plus_v,
                beta_65_plus_biv,
                beta_65_plus_ruv,
                beta_65_plus_rv,
                beta_65_plus_rbiv,
                delta_uv,
                delta_v,
                delta_biv,
                delta_5_17_uv,
                delta_5_17_v,
                delta_5_17_biv,
                delta_18_49_uv,
                delta_18_49_v,
                delta_18_49_biv,
                delta_50_64_uv,
                delta_50_64_v,
                delta_50_64_biv,
                delta_65_plus_uv,
                delta_65_plus_v,
                delta_65_plus_biv,
                gamma_i_uv,
                gamma_i_v,
                gamma_i_biv,
                gamma_h_uv,
                gamma_h_v,
                gamma_h_biv,
                gamma_i_5_17_uv,
                gamma_i_5_17_v,
                gamma_i_5_17_biv,
                gamma_h_5_17_uv,
                gamma_h_5_17_v,
                gamma_h_5_17_biv,
                gamma_i_18_49_uv,
                gamma_i_18_49_v,
                gamma_i_18_49_biv,
                gamma_h_18_49_uv,
                gamma_h_18_49_v,
                gamma_h_18_49_biv,
                gamma_i_50_64_uv,
                gamma_i_50_64_v,
                gamma_i_50_64_biv,
                gamma_h_50_64_uv,
                gamma_h_50_64_v,
                gamma_h_50_64_biv,
                gamma_i_65_plus_uv,
                gamma_i_65_plus_v,
                gamma_i_65_plus_biv,
                gamma_h_65_plus_uv,
                gamma_h_65_plus_v,
                gamma_h_65_plus_biv,
                mu_i_uv,
                mu_i_v,
                mu_i_biv,
                mu_h_uv,
                mu_h_v,
                mu_h_biv,
                mu_i_5_17_uv,
                mu_i_5_17_v,
                mu_i_5_17_biv,
                mu_h_5_17_uv,
                mu_h_5_17_v,
                mu_h_5_17_biv,
                mu_i_18_49_uv,
                mu_i_18_49_v,
                mu_i_18_49_biv,
                mu_h_18_49_uv,
                mu_h_18_49_v,
                mu_h_18_49_biv,
                mu_i_50_64_uv,
                mu_i_50_64_v,
                mu_i_50_64_biv,
                mu_h_50_64_uv,
                mu_h_50_64_v,
                mu_h_50_64_biv,
                mu_i_65_plus_uv,
                mu_i_65_plus_v,
                mu_i_65_plus_biv,
                mu_h_65_plus_uv,
                mu_h_65_plus_v,
                mu_h_65_plus_biv,
            ) = [
                ((parameter_value[index_param_previous_year] * 0.1 +
                  parameter_value[index_param] * 0.9)) - 0.025 * ((parameter_value[index_param_previous_year] * 0.1 +
                  parameter_value[index_param] * 0.9))

                # parameter_value[index_param]

                # hmean([parameter_value[index_param_previous_year],
                #        parameter_value[index_param_last_28]])
                for i, parameter_value in enumerate(parameter_values)
            ]

            # Scenario Assessment:
            multiplier = 0.75
            beta_uv *= multiplier
            beta_v *= multiplier
            beta_biv *= multiplier
            beta_ruv *= multiplier
            beta_rv *= multiplier
            beta_rbiv *= multiplier
            beta_5_17_uv *= multiplier
            beta_5_17_v *= multiplier
            beta_5_17_biv *= multiplier
            beta_5_17_ruv *= multiplier
            beta_5_17_rv *= multiplier
            beta_5_17_rbiv *= multiplier
            beta_18_49_uv *= multiplier
            beta_18_49_v *= multiplier
            beta_18_49_biv *= multiplier
            beta_18_49_ruv *= multiplier
            beta_18_49_rv *= multiplier
            beta_18_49_rbiv *= multiplier
            beta_50_64_uv *= multiplier
            beta_50_64_v *= multiplier
            beta_50_64_biv *= multiplier
            beta_50_64_ruv *= multiplier
            beta_50_64_rv *= multiplier
            beta_50_64_rbiv *= multiplier
            beta_65_plus_uv *= multiplier
            beta_65_plus_v *= multiplier
            beta_65_plus_biv *= multiplier
            beta_65_plus_ruv *= multiplier
            beta_65_plus_rv *= multiplier
            beta_65_plus_rbiv *= multiplier


            # HARMONIC MEAN:
            # harmonic_mean = (
            #     2
            #     * np.multiply(
            #         np.asarray(
            #             [
            #                 parameter_value[index_param_last_28]
            #                 for parameter_value in parameter_values
            #             ]
            #         ),
            #         np.asarray(
            #             [
            #                 parameter_value[index_param_previous_year]
            #                 for parameter_value in parameter_values
            #             ]
            #         ),
            #     )
            #     / (
            #         np.asarray(
            #             [
            #                 parameter_value[index_param_last_28]
            #                 for parameter_value in parameter_values
            #             ]
            #         )
            #         + np.asarray(
            #             [
            #                 parameter_value[index_param_previous_year]
            #                 for parameter_value in parameter_values
            #             ]
            #         )
            #     )
            # )

            # print("Parameter Index:", index_param)
            # print(alpha)
            # print(beta_uv)
            # print(beta_v)
            # parameter_values = list(self.epidemiological_model_parameters[state].values())
            #
            # print(parameter_values)
            # sys.exit()

            # Ordinary Differential Equations.
            index = (
                    int(timestep)
                    + len(self.epidemiological_model_data[state])
                    - len(simulation_data)
            )
            # print("vaccination index:", index)

            # Force of infection
            total_infections = max((i_uv + i_v + i_biv), 1)
            total_mobility = 1

            # Susceptible
            ds_uv_dt = s_uv + (
                    -beta_uv
                    * total_mobility
                    * s_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_uv
            )
            ds_v_dt = s_v + (
                    -beta_v
                    * total_mobility
                    * s_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_v
            )
            ds_biv_dt = s_biv + (
                    -beta_biv
                    * total_mobility
                    * s_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_v
            )

            ds_5_17_uv_dt = s_5_17_uv + (
                    -beta_5_17_uv
                    * total_mobility
                    * s_5_17_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_5_17_uv
            )
            ds_5_17_v_dt = s_5_17_v + (
                    -beta_5_17_v
                    * total_mobility
                    * s_5_17_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_5_17_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_5_17_v
            )
            ds_5_17_biv_dt = s_5_17_biv + (
                    -beta_5_17_biv
                    * total_mobility
                    * s_5_17_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_5_17_v
            )

            ds_18_49_uv_dt = s_18_49_uv + (
                    -beta_18_49_uv
                    * total_mobility
                    * s_18_49_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_18_49_uv
            )
            ds_18_49_v_dt = s_18_49_v + (
                    -beta_18_49_v
                    * total_mobility
                    * s_18_49_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_18_49_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_18_49_v
            )
            ds_18_49_biv_dt = s_18_49_biv + (
                    -beta_18_49_biv
                    * total_mobility
                    * s_18_49_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_18_49_v
            )

            ds_50_64_uv_dt = s_50_64_uv + (
                    -beta_50_64_uv
                    * total_mobility
                    * s_50_64_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_50_64_uv
            )
            ds_50_64_v_dt = s_50_64_v + (
                    -beta_50_64_v
                    * total_mobility
                    * s_50_64_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_50_64_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_50_64_v
            )
            ds_50_64_biv_dt = s_50_64_biv + (
                    -beta_50_64_biv
                    * total_mobility
                    * s_50_64_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_50_64_v
            )

            ds_65_plus_uv_dt = s_65_plus_uv + (
                    -beta_65_plus_uv
                    * total_mobility
                    * s_65_plus_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_65_plus_uv
            )
            ds_65_plus_v_dt = s_65_plus_v + (
                    -beta_65_plus_v
                    * total_mobility
                    * s_65_plus_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_65_plus_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_65_plus_v
            )
            ds_65_plus_biv_dt = s_65_plus_biv + (
                    -beta_65_plus_biv
                    * total_mobility
                    * s_65_plus_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * s_65_plus_v
            )

            # Infected
            di_uv_dt = i_uv + (
                    beta_uv
                    * total_mobility
                    * s_uv
                    * (total_infections ** alpha)
                    / population
                    + beta_ruv
                    * total_mobility
                    * r_uv
                    * (total_infections ** alpha)
                    / population
                    - delta_uv * i_uv
                    - gamma_i_uv * i_uv
                    - mu_i_uv * i_uv
            )
            di_v_dt = i_v + (
                    beta_v * total_mobility * s_v * (total_infections ** alpha) / population
                    + beta_rv
                    * total_mobility
                    * r_v
                    * (total_infections ** alpha)
                    / population
                    - delta_v * i_v
                    - gamma_i_v * i_v
                    - mu_i_v * i_v
            )
            di_biv_dt = i_biv + (
                    beta_biv
                    * total_mobility
                    * s_biv
                    * (total_infections ** alpha)
                    / population
                    + beta_rbiv
                    * total_mobility
                    * r_biv
                    * (total_infections ** alpha)
                    / population
                    - delta_biv * i_biv
                    - gamma_i_biv * i_biv
                    - mu_i_biv * i_biv
            )

            di_5_17_uv_dt = i_5_17_uv + (
                    beta_5_17_uv
                    * total_mobility
                    * s_5_17_uv
                    * (total_infections ** alpha)
                    / population
                    + beta_5_17_ruv
                    * total_mobility
                    * r_5_17_uv
                    * (total_infections ** alpha)
                    / population
                    - delta_5_17_uv * i_5_17_uv
                    - gamma_i_5_17_uv * i_5_17_uv
                    - mu_i_5_17_uv * i_5_17_uv
            )
            di_5_17_v_dt = i_5_17_v + (
                    beta_5_17_v
                    * total_mobility
                    * s_5_17_v
                    * (total_infections ** alpha)
                    / population
                    + beta_5_17_rv
                    * total_mobility
                    * r_5_17_v
                    * (total_infections ** alpha)
                    / population
                    - delta_5_17_v * i_5_17_v
                    - gamma_i_5_17_v * i_5_17_v
                    - mu_i_5_17_v * i_5_17_v
            )
            di_5_17_biv_dt = i_5_17_biv + (
                    beta_5_17_biv
                    * total_mobility
                    * s_5_17_biv
                    * (total_infections ** alpha)
                    / population
                    + beta_5_17_rbiv
                    * total_mobility
                    * r_5_17_biv
                    * (total_infections ** alpha)
                    / population
                    - delta_5_17_biv * i_5_17_biv
                    - gamma_i_5_17_biv * i_5_17_biv
                    - mu_i_5_17_biv * i_5_17_biv
            )

            di_18_49_uv_dt = i_18_49_uv + (
                    beta_18_49_uv
                    * total_mobility
                    * s_18_49_uv
                    * (total_infections ** alpha)
                    / population
                    + beta_18_49_ruv
                    * total_mobility
                    * r_18_49_uv
                    * (total_infections ** alpha)
                    / population
                    - delta_18_49_uv * i_18_49_uv
                    - gamma_i_18_49_uv * i_18_49_uv
                    - mu_i_18_49_uv * i_18_49_uv
            )
            di_18_49_v_dt = i_18_49_v + (
                    beta_18_49_v
                    * total_mobility
                    * s_18_49_v
                    * (total_infections ** alpha)
                    / population
                    + beta_18_49_rv
                    * total_mobility
                    * r_18_49_v
                    * (total_infections ** alpha)
                    / population
                    - delta_18_49_v * i_18_49_v
                    - gamma_i_18_49_v * i_18_49_v
                    - mu_i_18_49_v * i_18_49_v
            )
            di_18_49_biv_dt = i_18_49_biv + (
                    beta_18_49_biv
                    * total_mobility
                    * s_18_49_biv
                    * (total_infections ** alpha)
                    / population
                    + beta_18_49_rbiv
                    * total_mobility
                    * r_18_49_biv
                    * (total_infections ** alpha)
                    / population
                    - delta_18_49_biv * i_18_49_biv
                    - gamma_i_18_49_biv * i_18_49_biv
                    - mu_i_18_49_biv * i_18_49_biv
            )

            di_50_64_uv_dt = i_50_64_uv + (
                    beta_50_64_uv
                    * total_mobility
                    * s_50_64_uv
                    * (total_infections ** alpha)
                    / population
                    + beta_50_64_ruv
                    * total_mobility
                    * r_50_64_uv
                    * (total_infections ** alpha)
                    / population
                    - delta_50_64_uv * i_50_64_uv
                    - gamma_i_50_64_uv * i_50_64_uv
                    - mu_i_50_64_uv * i_50_64_uv
            )
            di_50_64_v_dt = i_50_64_v + (
                    beta_50_64_v
                    * total_mobility
                    * s_50_64_v
                    * (total_infections ** alpha)
                    / population
                    + beta_50_64_rv
                    * total_mobility
                    * r_50_64_v
                    * (total_infections ** alpha)
                    / population
                    - delta_50_64_v * i_50_64_v
                    - gamma_i_50_64_v * i_50_64_v
                    - mu_i_50_64_v * i_50_64_v
            )
            di_50_64_biv_dt = i_50_64_biv + (
                    beta_50_64_biv
                    * total_mobility
                    * s_50_64_biv
                    * (total_infections ** alpha)
                    / population
                    + beta_50_64_rbiv
                    * total_mobility
                    * r_50_64_biv
                    * (total_infections ** alpha)
                    / population
                    - delta_50_64_biv * i_50_64_biv
                    - gamma_i_50_64_biv * i_50_64_biv
                    - mu_i_50_64_biv * i_50_64_biv
            )

            di_65_plus_uv_dt = i_65_plus_uv + (
                    beta_65_plus_uv
                    * total_mobility
                    * s_65_plus_uv
                    * (total_infections ** alpha)
                    / population
                    + beta_65_plus_ruv
                    * total_mobility
                    * r_65_plus_uv
                    * (total_infections ** alpha)
                    / population
                    - delta_65_plus_uv * i_65_plus_uv
                    - gamma_i_65_plus_uv * i_65_plus_uv
                    - mu_i_65_plus_uv * i_65_plus_uv
            )
            di_65_plus_v_dt = i_65_plus_v + (
                    beta_65_plus_v
                    * total_mobility
                    * s_65_plus_v
                    * (total_infections ** alpha)
                    / population
                    + beta_65_plus_rv
                    * total_mobility
                    * r_65_plus_v
                    * (total_infections ** alpha)
                    / population
                    - delta_65_plus_v * i_65_plus_v
                    - gamma_i_65_plus_v * i_65_plus_v
                    - mu_i_65_plus_v * i_65_plus_v
            )
            di_65_plus_biv_dt = i_65_plus_biv + (
                    beta_65_plus_biv
                    * total_mobility
                    * s_65_plus_biv
                    * (total_infections ** alpha)
                    / population
                    + beta_65_plus_rbiv
                    * total_mobility
                    * r_65_plus_biv
                    * (total_infections ** alpha)
                    / population
                    - delta_65_plus_biv * i_65_plus_biv
                    - gamma_i_65_plus_biv * i_65_plus_biv
                    - mu_i_65_plus_biv * i_65_plus_biv
            )

            # Hospitalized
            dh_uv_dt = h_uv + delta_uv * i_uv - gamma_h_uv * h_uv - mu_h_uv * h_uv
            dh_v_dt = h_v + delta_v * i_v - gamma_h_v * h_v - mu_h_v * h_v
            dh_biv_dt = (
                    h_biv + delta_biv * i_biv - gamma_h_biv * h_biv - mu_h_biv * h_biv
            )

            dh_5_17_uv_dt = h_5_17_uv + (
                    delta_5_17_uv * i_5_17_uv
                    - gamma_h_5_17_uv * h_5_17_uv
                    - mu_h_5_17_uv * h_5_17_uv
            )
            dh_5_17_v_dt = h_5_17_v + (
                    delta_5_17_v * i_5_17_v
                    - gamma_h_5_17_v * h_5_17_v
                    - mu_h_5_17_v * h_5_17_v
            )
            dh_5_17_biv_dt = h_5_17_biv + (
                    delta_5_17_biv * i_5_17_biv
                    - gamma_h_5_17_biv * h_5_17_biv
                    - mu_h_5_17_biv * h_5_17_biv
            )

            dh_18_49_uv_dt = h_18_49_uv + (
                    delta_18_49_uv * i_18_49_uv
                    - gamma_h_18_49_uv * h_18_49_uv
                    - mu_h_18_49_uv * h_18_49_uv
            )
            dh_18_49_v_dt = h_18_49_v + (
                    delta_18_49_v * i_18_49_v
                    - gamma_h_18_49_v * h_18_49_v
                    - mu_h_18_49_v * h_18_49_v
            )
            dh_18_49_biv_dt = h_18_49_biv + (
                    delta_18_49_biv * i_18_49_biv
                    - gamma_h_18_49_biv * h_18_49_biv
                    - mu_h_18_49_biv * h_18_49_biv
            )

            dh_50_64_uv_dt = h_50_64_uv + (
                    delta_50_64_uv * i_50_64_uv
                    - gamma_h_50_64_uv * h_50_64_uv
                    - mu_h_50_64_uv * h_50_64_uv
            )
            dh_50_64_v_dt = h_50_64_v + (
                    delta_50_64_v * i_50_64_v
                    - gamma_h_50_64_v * h_50_64_v
                    - mu_h_50_64_v * h_50_64_v
            )
            dh_50_64_biv_dt = h_50_64_biv + (
                    delta_50_64_biv * i_50_64_biv
                    - gamma_h_50_64_biv * h_50_64_biv
                    - mu_h_50_64_biv * h_50_64_biv
            )

            dh_65_plus_uv_dt = h_65_plus_uv + (
                    delta_65_plus_uv * i_65_plus_uv
                    - gamma_h_65_plus_uv * h_65_plus_uv
                    - mu_h_65_plus_uv * h_65_plus_uv
            )
            dh_65_plus_v_dt = h_65_plus_v + (
                    delta_65_plus_v * i_65_plus_v
                    - gamma_h_65_plus_v * h_65_plus_v
                    - mu_h_65_plus_v * h_65_plus_v
            )
            dh_65_plus_biv_dt = h_65_plus_biv + (
                    delta_65_plus_biv * i_65_plus_biv
                    - gamma_h_65_plus_biv * h_65_plus_biv
                    - mu_h_65_plus_biv * h_65_plus_biv
            )

            # Recovered
            dr_uv_dt = r_uv + (
                    gamma_i_uv * i_uv
                    + gamma_h_uv * h_uv
                    - beta_ruv
                    * total_mobility
                    * r_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_uv
            )
            dr_v_dt = r_v + (
                    gamma_i_v * i_v
                    + gamma_h_v * h_v
                    - beta_rv
                    * total_mobility
                    * r_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_v
            )
            dr_biv_dt = r_biv + (
                    gamma_i_biv * i_biv
                    + gamma_h_biv * h_biv
                    - beta_rbiv
                    * total_mobility
                    * r_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_v
            )

            dr_5_17_uv_dt = r_5_17_uv + (
                    gamma_i_5_17_uv * i_5_17_uv
                    + gamma_h_5_17_uv * h_5_17_uv
                    - beta_5_17_ruv
                    * total_mobility
                    * r_5_17_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_5_17_uv
            )
            dr_5_17_v_dt = r_5_17_v + (
                    gamma_i_5_17_v * i_5_17_v
                    + gamma_h_5_17_v * h_5_17_v
                    - beta_5_17_rv
                    * total_mobility
                    * r_5_17_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_5_17_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_5_17_v
            )
            dr_5_17_biv_dt = r_5_17_biv + (
                    gamma_i_5_17_biv * i_5_17_biv
                    + gamma_h_5_17_biv * h_5_17_biv
                    - beta_5_17_rbiv
                    * total_mobility
                    * r_5_17_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_5_17_v
            )

            dr_18_49_uv_dt = r_18_49_uv + (
                    gamma_i_18_49_uv * i_18_49_uv
                    + gamma_h_18_49_uv * h_18_49_uv
                    - beta_18_49_ruv
                    * total_mobility
                    * r_18_49_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_18_49_uv
            )
            dr_18_49_v_dt = r_18_49_v + (
                    gamma_i_18_49_v * i_18_49_v
                    + gamma_h_18_49_v * h_18_49_v
                    - beta_18_49_rv
                    * total_mobility
                    * r_18_49_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_18_49_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_18_49_v
            )
            dr_18_49_biv_dt = r_18_49_biv + (
                    gamma_i_18_49_biv * i_18_49_biv
                    + gamma_h_18_49_biv * h_18_49_biv
                    - beta_18_49_rbiv
                    * total_mobility
                    * r_18_49_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_18_49_v
            )

            dr_50_64_uv_dt = r_50_64_uv + (
                    gamma_i_50_64_uv * i_50_64_uv
                    + gamma_h_50_64_uv * h_50_64_uv
                    - beta_50_64_ruv
                    * total_mobility
                    * r_50_64_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_50_64_uv
            )
            dr_50_64_v_dt = r_50_64_v + (
                    gamma_i_50_64_v * i_50_64_v
                    + gamma_h_50_64_v * h_50_64_v
                    - beta_50_64_rv
                    * total_mobility
                    * r_50_64_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_50_64_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_50_64_v
            )
            dr_50_64_biv_dt = r_50_64_biv + (
                    gamma_i_50_64_biv * i_50_64_biv
                    + gamma_h_50_64_biv * h_50_64_biv
                    - beta_50_64_rbiv
                    * total_mobility
                    * r_50_64_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_50_64_v
            )

            dr_65_plus_uv_dt = r_65_plus_uv + (
                    gamma_i_65_plus_uv * i_65_plus_uv
                    + gamma_h_65_plus_uv * h_65_plus_uv
                    - beta_65_plus_ruv
                    * total_mobility
                    * r_65_plus_uv
                    * (total_infections ** alpha)
                    / population
                    - self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_65_plus_uv
            )
            dr_65_plus_v_dt = r_65_plus_v + (
                    gamma_i_65_plus_v * i_65_plus_v
                    + gamma_h_65_plus_v * h_65_plus_v
                    - beta_65_plus_rv
                    * total_mobility
                    * r_65_plus_v
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_unvaccinated_to_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_65_plus_uv
                    - self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_65_plus_v
            )
            dr_65_plus_biv_dt = r_65_plus_biv + (
                    gamma_i_65_plus_biv * i_65_plus_biv
                    + gamma_h_65_plus_biv * h_65_plus_biv
                    - beta_65_plus_rbiv
                    * total_mobility
                    * r_65_plus_biv
                    * (total_infections ** alpha)
                    / population
                    + self.epidemiological_model_data[state][
                        "percentage_vaccinated_to_bivalent_vaccinated"
                    ].iloc[min(int(index), len(self.epidemiological_model_data[state]) - 1)]
                    * r_65_plus_v
            )
            # Deceased
            ddec_uv_dt = dec_uv + mu_i_uv * i_uv + mu_h_uv * h_uv
            ddec_v_dt = dec_v + mu_i_v * i_v + mu_h_v * h_v
            ddec_biv_dt = dec_biv + mu_i_biv * i_biv + mu_h_biv * h_biv

            ddec_5_17_uv_dt = (
                    dec_5_17_uv + mu_i_5_17_uv * i_5_17_uv + mu_h_5_17_uv * h_5_17_uv
            )
            ddec_5_17_v_dt = (
                    dec_5_17_v + mu_i_5_17_v * i_5_17_v + mu_h_5_17_v * h_5_17_v
            )
            ddec_5_17_biv_dt = (
                    dec_5_17_biv + mu_i_5_17_biv * i_5_17_biv + mu_h_5_17_biv * h_5_17_biv
            )

            ddec_18_49_uv_dt = (
                    dec_18_49_uv + mu_i_18_49_uv * i_18_49_uv + mu_h_18_49_uv * h_18_49_uv
            )
            ddec_18_49_v_dt = (
                    dec_18_49_v + mu_i_18_49_v * i_18_49_v + mu_h_18_49_v * h_18_49_v
            )
            ddec_18_49_biv_dt = dec_18_49_biv + (
                    mu_i_18_49_biv * i_18_49_biv + mu_h_18_49_biv * h_18_49_biv
            )

            ddec_50_64_uv_dt = (
                    dec_50_64_uv + mu_i_50_64_uv * i_50_64_uv + mu_h_50_64_uv * h_50_64_uv
            )
            ddec_50_64_v_dt = (
                    dec_50_64_v + mu_i_50_64_v * i_50_64_v + mu_h_50_64_v * h_50_64_v
            )
            ddec_50_64_biv_dt = dec_50_64_biv + (
                    mu_i_50_64_biv * i_50_64_biv + mu_h_50_64_biv * h_50_64_biv
            )

            ddec_65_plus_uv_dt = dec_65_plus_uv + (
                    mu_i_65_plus_uv * i_65_plus_uv + mu_h_65_plus_uv * h_65_plus_uv
            )
            ddec_65_plus_v_dt = dec_65_plus_v + (
                    mu_i_65_plus_v * i_65_plus_v + mu_h_65_plus_v * h_65_plus_v
            )
            ddec_65_plus_biv_dt = dec_65_plus_biv + (
                    mu_i_65_plus_biv * i_65_plus_biv + mu_h_65_plus_biv * h_65_plus_biv
            )

            # Update

            # Updated values:
            updated_values = (
                ds_uv_dt,
                ds_v_dt,
                ds_biv_dt,
                di_uv_dt,
                di_v_dt,
                di_biv_dt,
                dh_uv_dt,
                dh_v_dt,
                dh_biv_dt,
                dr_uv_dt,
                dr_v_dt,
                dr_biv_dt,
                ddec_uv_dt,
                ddec_v_dt,
                ddec_biv_dt,
                ds_5_17_uv_dt,
                ds_5_17_v_dt,
                ds_5_17_biv_dt,
                ds_18_49_uv_dt,
                ds_18_49_v_dt,
                ds_18_49_biv_dt,
                ds_50_64_uv_dt,
                ds_50_64_v_dt,
                ds_50_64_biv_dt,
                ds_65_plus_uv_dt,
                ds_65_plus_v_dt,
                ds_65_plus_biv_dt,
                di_5_17_uv_dt,
                di_5_17_v_dt,
                di_5_17_biv_dt,
                di_18_49_uv_dt,
                di_18_49_v_dt,
                di_18_49_biv_dt,
                di_50_64_uv_dt,
                di_50_64_v_dt,
                di_50_64_biv_dt,
                di_65_plus_uv_dt,
                di_65_plus_v_dt,
                di_65_plus_biv_dt,
                dh_5_17_uv_dt,
                dh_5_17_v_dt,
                dh_5_17_biv_dt,
                dh_18_49_uv_dt,
                dh_18_49_v_dt,
                dh_18_49_biv_dt,
                dh_50_64_uv_dt,
                dh_50_64_v_dt,
                dh_50_64_biv_dt,
                dh_65_plus_uv_dt,
                dh_65_plus_v_dt,
                dh_65_plus_biv_dt,
                dr_5_17_uv_dt,
                dr_5_17_v_dt,
                dr_5_17_biv_dt,
                dr_18_49_uv_dt,
                dr_18_49_v_dt,
                dr_18_49_biv_dt,
                dr_50_64_uv_dt,
                dr_50_64_v_dt,
                dr_50_64_biv_dt,
                dr_65_plus_uv_dt,
                dr_65_plus_v_dt,
                dr_65_plus_biv_dt,
                ddec_5_17_uv_dt,
                ddec_5_17_v_dt,
                ddec_5_17_biv_dt,
                ddec_18_49_uv_dt,
                ddec_18_49_v_dt,
                ddec_18_49_biv_dt,
                ddec_50_64_uv_dt,
                ddec_50_64_v_dt,
                ddec_50_64_biv_dt,
                ddec_65_plus_uv_dt,
                ddec_65_plus_v_dt,
                ddec_65_plus_biv_dt,
            )

            (
                s_uv,
                s_v,
                s_biv,
                i_uv,
                i_v,
                i_biv,
                h_uv,
                h_v,
                h_biv,
                r_uv,
                r_v,
                r_biv,
                dec_uv,
                dec_v,
                dec_biv,
                s_5_17_uv,
                s_5_17_v,
                s_5_17_biv,
                s_18_49_uv,
                s_18_49_v,
                s_18_49_biv,
                s_50_64_uv,
                s_50_64_v,
                s_50_64_biv,
                s_65_plus_uv,
                s_65_plus_v,
                s_65_plus_biv,
                i_5_17_uv,
                i_5_17_v,
                i_5_17_biv,
                i_18_49_uv,
                i_18_49_v,
                i_18_49_biv,
                i_50_64_uv,
                i_50_64_v,
                i_50_64_biv,
                i_65_plus_uv,
                i_65_plus_v,
                i_65_plus_biv,
                h_5_17_uv,
                h_5_17_v,
                h_5_17_biv,
                h_18_49_uv,
                h_18_49_v,
                h_18_49_biv,
                h_50_64_uv,
                h_50_64_v,
                h_50_64_biv,
                h_65_plus_uv,
                h_65_plus_v,
                h_65_plus_biv,
                r_5_17_uv,
                r_5_17_v,
                r_5_17_biv,
                r_18_49_uv,
                r_18_49_v,
                r_18_49_biv,
                r_50_64_uv,
                r_50_64_v,
                r_50_64_biv,
                r_65_plus_uv,
                r_65_plus_v,
                r_65_plus_biv,
                dec_5_17_uv,
                dec_5_17_v,
                dec_5_17_biv,
                dec_18_49_uv,
                dec_18_49_v,
                dec_18_49_biv,
                dec_50_64_uv,
                dec_50_64_v,
                dec_50_64_biv,
                dec_65_plus_uv,
                dec_65_plus_v,
                dec_65_plus_biv,
            ) = updated_values

            # print(
            #     "Before:\n",
            #     self.simulation_data[state],
            # )
            new_row = {
                "date": [
                    pd.to_datetime(
                        self.population_dynamics_computer_configuration[
                            "simulation_start_date"
                        ]
                    )
                    + DateOffset(days=timestep)
                ],
            }

            for i in range(len(self.epidemiological_compartment_names)):
                new_row[
                    f"{self.epidemiological_compartment_names[i]}"
                ] = updated_values[i]

            self.simulation_data[state] = pd.concat(
                [self.simulation_data[state], pd.DataFrame(new_row)],
                ignore_index=True,
            )
            # print(
            #     "\n\nAfter:\n",
            #     self.simulation_data[state],
            # )

        self.simulation_data[state].to_csv(
            f"{data_directory}/epidemic_forecasts/scenario_assessment/{state}.csv",
            index=False,
        )

        self.plot(
            state=state,
            actual_values=simulation_data[
                self.epidemiological_compartment_names
            ].values,
            model_predictions=self.simulation_data[state][
                self.epidemiological_compartment_names
            ].values,
        )
        # print("Actual Values Shape:", simulation_data[
        #     self.epidemiological_compartment_names
        # ].values.shape)
        # print("Predictions Shape:", self.simulation_data[state][
        #     self.epidemiological_compartment_names
        # ].values.shape)
        mape = mean_absolute_percentage_error(simulation_data[
                                                  self.epidemiological_compartment_names
                                              ].values + 1, self.simulation_data[state][
                                                  self.epidemiological_compartment_names
                                              ].values) * 100
        smape = sMAPE(simulation_data[self.epidemiological_compartment_names].values, self.simulation_data[state][
            self.epidemiological_compartment_names
        ].values) * 100
        rmse = mean_squared_error(simulation_data[self.epidemiological_compartment_names].values,
                                  self.simulation_data[state][
                                      self.epidemiological_compartment_names
                                  ].values, squared=False)
        self.average_mape.append(mape)
        self.average_smape.append(smape)
        self.average_rmse.append(rmse)

        # print(mape, smape, rmse, self.average_mape, self.average_smape, self.average_rmse)

    def plot(self, state, actual_values, model_predictions):
        """This method plots the model predictions vs the actual data.

        :parameter model_predictions: Array - Model predictions."""

        for i, compartment_name in enumerate(self.epidemiological_compartment_names):
            if not os.path.exists(
                    f"{data_directory}/epidemic_forecasts/scenario_assessment/{state}"
            ):
                os.makedirs(
                    f"{data_directory}/epidemic_forecasts/scenario_assessment/{state}"
                )

            plt.figure(figsize=(16, 10))
            plt.plot(
                actual_values[:, i],
                linewidth=3,
                label=compartment_name,
            )
            plt.plot(
                model_predictions[:, i],
                "--",
                linewidth=3,
                c="red",
                label="Best Fit ODE",
            )
            plt.xlabel("Days", fontsize=24)
            plt.ylabel("Population", fontsize=24)
            plt.title(f"{compartment_name} vs Best Fit ODE", fontsize=32)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
            plt.grid()
            # plt.savefig(
            #     f"{data_directory}/epidemic_forecasts/scenario_assessment/{state}/{compartment_name}.png"
            # )
            plt.close()
            # plt.show()


if __name__ == "__main__":
    epidemiological_compartments = []
    age_groups = ["5-17", "18-49", "50-64", "65+"]
    vaccination_groups = ["UV", "V", "BiV"]
    compartments = [
        "Susceptible",
        "Infected",
        "Hospitalized",
        "Recovered",
        "Deceased",
    ]

    for compartment in compartments:
        for vaccination_group in vaccination_groups:
            epidemiological_compartments.append(f"{compartment}_{vaccination_group}")

    for compartment in compartments:
        for age_group in age_groups:
            for vaccination_group in vaccination_groups:
                epidemiological_compartments.append(
                    f"{compartment}_{age_group}_{vaccination_group}"
                )

    pd_computer_configuration = {
        "data_path": f"{data_directory}/epidemiological_model_data/",
        "output_path": f"{data_directory}/epidemic_forecasts/",
        "simulation_start_date": "01/01/2023",
        "epidemiological_compartment_names": epidemiological_compartments,
        "parameter_computation_timeframe": 28,
    }

    epidemic_simulator = PopulationDynamicsComputer(
        population_dynamics_computer_configuration=pd_computer_configuration
    )
    parameter_initializer = ParameterInitializer(
        data_path=f"{data_directory}/epidemiological_model_data/"
    )
    states = parameter_initializer.initialize_state_names()

    # Parallel Computing:
    # pool = Pool(24)
    # pool.map(epidemic_simulator.epidemic_forecasting, states)
    # epidemic_simulator.epidemic_forecasting(state="New York")
    epidemic_simulator.epidemic_forecasting(state="Pennsylvania")
    # for state in states:
    # epidemic_simulator.epidemic_forecasting(state="USA")
    print(f"Average MAPE: {epidemic_simulator.average_mape} %")
    print(f"Average SMAPE: {epidemic_simulator.average_smape} %")
    print(f"Average RMSE: {epidemic_simulator.average_rmse}")

    # epidemic_simulator.plot(
    #     state="USA",
    #     actual_values=pd.read_csv(f"{data_directory}/epidemiological_model_data/USA.csv")[
    #         epidemiological_compartments
    #     ]
    #     .iloc[-90:]
    #     .values,
    #     model_predictions=pd.read_csv(
    #         f"{data_directory}/epidemiological_model_parameters/model_predictions/USA.csv"
    #     )[epidemiological_compartments]
    #     .iloc[-90:]
    #     .values,
    # )
