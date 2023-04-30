import numpy as np
import pandas as pd


class PopulationDynamicsComputer:

    @staticmethod
    def compute_population_dynamics(action, beta, environment_config, epidemiological_model_data,
                                    epidemiological_model_parameters, new_cases, population_dynamics_dataframes, state,
                                    state_populations, timestep,
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
            percentage_unvaccinated_to_fully_vaccinated = (
                epidemiological_model_data[state][
                    "percentage_unvaccinated_to_fully_vaccinated"
                ].iloc[timestep + 214]
            )
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
            model_parameter_value = np.random.normal(mu, sigma, 1)
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
            population_dynamics_dataframes[state]["unvaccinated_individuals"].iloc[
                -1
            ]
            - percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state][
                "unvaccinated_individuals"
            ].iloc[-1]
        )

        number_of_fully_vaccinated_individuals = int(
            population_dynamics_dataframes[state][
                "fully_vaccinated_individuals"
            ].iloc[-1]
            + percentage_unvaccinated_to_fully_vaccinated
            * population_dynamics_dataframes[state][
                "unvaccinated_individuals"
            ].iloc[-1]
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

        return epidemiological_model_parameters, population_dynamics_dataframes, new_cases
