# Imports
import sys

import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from lmfit import minimize
from time import time
from src.settings import DATA_DIR
from src.utilities.parameter_initializer import ParameterInitializer


class EpidemiologicalModelParameterComputer:
    """This class computes the parameters for the epidemiological model."""

    def __init__(self, parameter_computer_configuration):
        """This method initializes the parameters for computing the epidemiological model parameters.

        :param parameter_computer_configuration: Dictionary containing the configuration for epidemiological model
                                                 parameter computation.
        """

        self.parameter_computer_configuration = parameter_computer_configuration

        self.parameter_initializer = ParameterInitializer(
            data_path=self.parameter_computer_configuration["data_path"]
        )

        self.states = self.parameter_initializer.initialize_state_names()
        # print("State Names:", self.states)

        self.epidemiological_model_data = (
            self.parameter_initializer.initialize_epidemiological_model_data()
        )
        # print("Epidemiological Model Data:\n", self.epidemiological_model_data)

        self.state_populations = (
            self.parameter_initializer.initialize_state_populations()
        )
        # print("\nState Populations:\n", self.state_populations)

        # self.epidemiological_data = pd.read_csv(data_path)
        # self.population = population
        self.epidemiological_model_parameters = self.parameter_initializer.initialize_initial_epidemiological_model_parameters(
            self.parameter_computer_configuration["constrained_beta"]
        )

        self.epidemiological_compartment_names = parameter_computer_configuration[
            "epidemiological_compartment_names"
        ]

        # self.epidemiological_data = self.epidemiological_data.iloc[
        #                             data_split_min:data_split_max
        #                             ]
        #
        # self.epidemiological_compartment_values = self.epidemiological_data[
        #     self.epidemiological_compartment_names
        # ].values
        # self.t = np.linspace(
        #     0,
        #     self.epidemiological_data.shape[0] - 1,
        #     self.epidemiological_data.shape[0],
        # )

        # Initial values of population dynamics.
        # self.y0 = [
        #     self.epidemiological_model_data[self.states[0]][
        #         f"{self.epidemiological_compartment_names[i]}"
        #     ].iloc[0]
        #     for i in range(len(self.epidemiological_compartment_names))
        # ]

        self.original_residual = None

        self.counter = 1

    def compute_epidemiological_model_parameters(self):
        """This method computes the epidemiological model parameters and saves them..."""

        parameter_computation_timeframe = self.parameter_computer_configuration[
            "parameter_computation_timeframe"
        ]

        for state in self.epidemiological_model_data:
            print("\nState:", state)
            state_runtime_start = time()

            number_of_splits = int(
                np.ceil(
                    self.epidemiological_model_data[state].shape[0]
                    / parameter_computation_timeframe
                )
            )

            model_predictions = []
            data_values = []
            original_residuals = []
            total_residual = 0
            param_dict = {}

            for split_number in range(number_of_splits):
                print(f"Split {split_number + 1} of {number_of_splits}:\n")

                split_min_index = split_number * parameter_computation_timeframe
                split_max_index = min(
                    (split_number + 1) * parameter_computation_timeframe,
                    self.epidemiological_model_data[state].shape[0],
                )

                split_epidemiological_data = self.epidemiological_model_data[
                    state
                ].iloc[split_min_index:split_max_index]

                epidemiological_model_compartment_values = split_epidemiological_data[
                    self.epidemiological_compartment_names
                ].values

                t = np.linspace(
                    0,
                    split_epidemiological_data.shape[0] - 1,
                    split_epidemiological_data.shape[0],
                )

                y0 = [
                    split_epidemiological_data[
                        f"{self.epidemiological_compartment_names[i]}"
                    ].iloc[0]
                    for i in range(len(self.epidemiological_compartment_names))
                ]

                split_runtime_start = time()

                model_fit_solve_ivp = minimize(
                    self.residual_solve_ivp,
                    self.epidemiological_model_parameters,
                    args=(
                        t,
                        epidemiological_model_compartment_values,
                        "RK45",
                        1,
                        state,
                        y0,
                    ),
                    method="leastsq",
                    nan_policy="omit",
                )

                print("Split Runtime:", time() - split_runtime_start, "seconds")

                model_pred = (
                    epidemiological_model_compartment_values
                    + self.original_residual.reshape(
                        epidemiological_model_compartment_values.shape
                    )
                )

                data = epidemiological_model_compartment_values

                model_predictions.append(model_pred)
                data_values.append(data)

                original_residuals.append(self.original_residual)

                print(
                    "Split Residual:",
                    np.sum(np.abs(self.original_residual)),
                )

                #     print(f'\n\nModel Fit {split_number}:\n', report_fit(model_fit_solve_ivp))
                #     print(model_fit_solve_ivp.params.pretty_print())
                print("-------------------------------")

                for name, param in model_fit_solve_ivp.params.items():
                    if name not in param_dict:
                        param_dict[f"{name}"] = [param.value]
                    else:
                        param_dict[f"{name}"].append(param.value)

            total_residual += np.sum(np.abs(np.asarray(original_residuals[:-1])))

            print("Total Residual:", total_residual)
            print("State Runtime:", time() - state_runtime_start, "seconds")

            for key in param_dict.keys():
                print(f"{key}:", param_dict[key])
            model_pred_overall = np.concatenate(
                [model_predictions[i] for i in range(len(model_predictions))]
            )
            data_overall = np.concatenate(
                [data_values[i] for i in range(len(data_values))]
            )

            self.plot(actual_values=data_overall, model_predictions=model_pred_overall)
            print("\n\n\n\n\n\n")

    def differential_equations(
        self,
        y,
        t,
        population,
        parameters,
        call_signature_ode_int=True,
        differential_equations_version=1,
        state="New York",
    ):
        """This method models the differential equations for the epidemiological model.

        :param state: Name of the state
        :param y: Vector of sub-compartment population dynamics
        :param t: Time span of simulation
        :param population: Total Population
        :param parameters: Parameter values
        :param call_signature_ode_int: Boolean - Indicates if the calling signature of the differential_equations method
                                                 is that of scipy's odeint method.
        :param differential_equations_version: Integer: Version of differential equations/model to be used.

        :returns derivatives of the model compartments."""

        if not call_signature_ode_int:
            t, y = y, t

        # VERSION 1:
        if differential_equations_version == 1:
            """SEIQRD Model with standard incidence that doesn't correct for deaths and hospitalizations.
            We allow recovered individuals to be susceptible again. Model accounts for vaccination rates.
            We model susceptible, exposed, and recovered people to be vaccinated. Exposure rate is
            dependent on the number of new cases."""

            # Sub-compartments
            (
                s_uv,
                s_fv,
                s_bv,
                e_uv,
                e_fv,
                e_bv,
                i_uv,
                i_fv,
                i_bv,
                h_uv,
                h_fv,
                h_bv,
                r_uv,
                r_fv,
                r_bv,
                d_uv,
                d_fv,
                d_bv,
            ) = y

            # Force of infection
            total_infections = max((i_uv + i_fv + i_bv), 1)

            # Parameter Values
            beta = parameters["beta"].value
            alpha = parameters["alpha"].value

            zeta_uv = parameters["zeta_uv"].value
            zeta_fv = parameters["zeta_fv"].value
            zeta_bv = parameters["zeta_bv"].value
            zeta_ruv = parameters["zeta_ruv"].value
            zeta_rfv = parameters["zeta_rfv"].value
            zeta_rbv = parameters["zeta_rbv"].value

            delta_uv = parameters["delta_uv"].value
            delta_fv = parameters["delta_fv"].value
            delta_bv = parameters["delta_bv"].value

            mu_i_uv = parameters["mu_i_uv"].value
            mu_i_fv = parameters["mu_i_fv"].value
            mu_i_bv = parameters["mu_i_bv"].value

            mu_h_uv = parameters["mu_h_uv"].value
            mu_h_fv = parameters["mu_h_fv"].value
            mu_h_bv = parameters["mu_h_bv"].value

            gamma_i_uv = parameters["gamma_i_uv"].value
            gamma_i_fv = parameters["gamma_i_fv"].value
            gamma_i_bv = parameters["gamma_i_bv"].value

            gamma_h_uv = parameters["gamma_h_uv"].value
            gamma_h_fv = parameters["gamma_h_fv"].value
            gamma_h_bv = parameters["gamma_h_bv"].value

            exp_to_s_uv = parameters["exp_to_suv"].value
            exp_to_s_fv = parameters["exp_to_sfv"].value
            exp_to_s_bv = parameters["exp_to_sbv"].value
            exp_to_r_uv = parameters["exp_to_ruv"].value
            exp_to_r_fv = parameters["exp_to_rfv"].value
            exp_to_r_bv = parameters["exp_to_rbv"].value

            # Ordinary Differential Equations.
            # beta = beta_s * self.epidemiological_model_data[state]['New Cases'].iloc[
            #     min(int(t), len(self.epidemiological_model_data[state]) - 1)]
            # beta = beta_s

            # Susceptible
            ds_uv_dt = (
                -beta * s_uv * (total_infections**alpha) / population
                + exp_to_s_uv * e_uv
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_fully_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_uv
            )

            ds_fv_dt = (
                -beta * s_fv * (total_infections**alpha) / population
                + exp_to_s_fv * e_fv
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_fully_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_uv
                - self.epidemiological_model_data[state][
                    "percentage_fully_vaccinated_to_boosted"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_fv
            )

            ds_bv_dt = (
                -beta * s_bv * (total_infections**alpha) / population
                + exp_to_s_bv * e_bv
                + self.epidemiological_model_data[state][
                    "percentage_fully_vaccinated_to_boosted"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_fv
            )

            # Exposed
            de_uv_dt = (
                beta * s_uv * (total_infections**alpha) / population
                + beta * r_uv * (total_infections**alpha) / population
                - zeta_uv * e_uv
                - zeta_ruv * e_uv
                - exp_to_s_uv * e_uv
                - exp_to_r_uv * e_uv
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_fully_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * e_uv
            )
            de_fv_dt = (
                beta * s_fv * (total_infections**alpha) / population
                + beta * r_fv * (total_infections**alpha) / population
                - zeta_fv * e_fv
                - zeta_rfv * e_fv
                - exp_to_s_fv * e_fv
                - exp_to_r_fv * e_fv
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_fully_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * e_uv
                - self.epidemiological_model_data[state][
                    "percentage_fully_vaccinated_to_boosted"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * e_fv
            )
            de_bv_dt = (
                beta * s_bv * (total_infections**alpha) / population
                + beta * r_bv * (total_infections**alpha) / population
                - zeta_bv * e_bv
                - zeta_rbv * e_bv
                - exp_to_s_bv * e_bv
                - exp_to_r_bv * e_bv
                + self.epidemiological_model_data[state][
                    "percentage_fully_vaccinated_to_boosted"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * e_fv
            )

            # Infected
            di_uv_dt = (
                zeta_uv * e_uv
                + zeta_ruv * e_uv
                - delta_uv * i_uv
                - gamma_i_uv * i_uv
                - mu_i_uv * i_uv
            )
            di_fv_dt = (
                zeta_fv * e_fv
                + zeta_rfv * e_fv
                - delta_fv * i_fv
                - gamma_i_fv * i_fv
                - mu_i_fv * i_fv
            )
            di_bv_dt = (
                zeta_bv * e_bv
                + zeta_rbv * e_bv
                - delta_bv * i_bv
                - gamma_i_bv * i_bv
                - mu_i_bv * i_bv
            )

            # Hospitalized
            dh_uv_dt = delta_uv * i_uv - gamma_h_uv * h_uv - mu_h_uv * h_uv
            dh_fv_dt = delta_fv * i_fv - gamma_h_fv * h_fv - mu_h_fv * h_fv
            dh_bv_dt = delta_bv * i_bv - gamma_h_bv * h_bv - mu_h_bv * h_bv

            # Recovered
            dr_uv_dt = (
                gamma_i_uv * i_uv
                + gamma_h_uv * h_uv
                - beta * r_uv * (total_infections**alpha) / population
                + exp_to_r_uv * e_uv
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_fully_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_uv
            )
            dr_fv_dt = (
                gamma_i_fv * i_fv
                + gamma_h_fv * h_fv
                - beta * r_fv * (total_infections**alpha) / population
                + exp_to_r_fv * e_fv
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_fully_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_uv
                - self.epidemiological_model_data[state][
                    "percentage_fully_vaccinated_to_boosted"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_fv
            )
            dr_bv_dt = (
                gamma_i_bv * i_bv
                + gamma_h_bv * h_bv
                - beta * r_bv * (total_infections**alpha) / population
                + exp_to_r_bv * e_bv
                + self.epidemiological_model_data[state][
                    "percentage_fully_vaccinated_to_boosted"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_fv
            )

            #             dr_uv_dt = gamma_i_uv * i_uv + gamma_h_uv * h_uv
            #             dr_fv_dt = gamma_i_bv * i_fv + gamma_h_bv * h_fv
            #             dr_bv_dt = gamma_i_bv * i_bv + gamma_h_bv * h_bv

            # Deceased
            dd_uv_dt = mu_i_uv * i_uv + mu_h_uv * h_uv
            dd_fv_dt = mu_i_fv * i_fv + mu_h_fv * h_fv
            dd_bv_dt = mu_i_bv * i_bv + mu_h_bv * h_bv

            return (
                ds_uv_dt,
                ds_fv_dt,
                ds_bv_dt,
                de_uv_dt,
                de_fv_dt,
                de_bv_dt,
                di_uv_dt,
                di_fv_dt,
                di_bv_dt,
                dh_uv_dt,
                dh_fv_dt,
                dh_bv_dt,
                dr_uv_dt,
                dr_fv_dt,
                dr_bv_dt,
                dd_uv_dt,
                dd_fv_dt,
                dd_bv_dt,
            )

    def ode_solver(
        self,
        y0,
        t,
        population,
        parameters,
        solver="odeint",
        method="RK45",
        differential_equations_version=1,
        state="New York",
    ):
        """This function solves the ordinary differential equations for the epidemiological model.

        :parameter y0: Vector of initial population dynamics
        :parameter t: Time span of simulation
        :parameter population: Total Population
        :parameter parameters: Parameter values
        :parameter solver: String - Name of the solver
        :parameter method: Integration method used by the solver.
        :parameter differential_equations_version: Integer representing the model/differential equations we want to use.

        :return x_odeint: model predictions from Scipy's odeint method
        :return x_solve_ivp.y.T: model predictions form Scipy's solve_ivp method"""

        if solver == "odeint":
            x_odeint = odeint(
                self.differential_equations,
                y0,
                t,
                args=(
                    population,
                    parameters,
                    True,
                    differential_equations_version,
                    state,
                ),
            )
            #             print(self.counter)
            #             self.counter += 1
            return x_odeint

        elif solver == "solve_ivp":
            x_solve_ivp = solve_ivp(
                self.differential_equations,
                y0=y0,
                t_span=(min(t), max(t)),
                t_eval=t,
                method=method,
                args=(
                    population,
                    parameters,
                    False,
                    differential_equations_version,
                    state,
                ),
            )
            #             print(self.counter)
            #             self.counter += 1
            return x_solve_ivp.y.T

    def residual(
        self,
        parameters,
        t,
        data,
        solver="odeint",
        method="RK45",
        differential_equations_version=1,
        state="New York",
        initial_values=None,
    ):
        """This function computes the residuals between the model predictions and the actual data.

        :parameter parameters: Parameter values
        :parameter t: Time span of simulation
        :parameter data: Real-world data we want to fit our model to
        :parameter solver: String - Name of the solver
        :parameter method: Integration method used by the solver.
        :parameter differential_equations_version: Integer representing the model/differential equations we want to use.
        :param state:
        :param initial_values:

        :returns: residuals"""

        model_predictions = self.ode_solver(
            initial_values,
            t,
            self.state_populations[state],
            parameters,
            solver=solver,
            method=method,
            differential_equations_version=differential_equations_version,
            state=state,
        )

        model_predictions = pd.DataFrame(
            model_predictions, columns=self.epidemiological_compartment_names
        )

        # The original way in which residuals were computed. Takes the difference between predictions and data.
        residual_original = (model_predictions.values - data).ravel()
        self.original_residual = residual_original

        # Normalized residuals so all features contribute equally to the loss.
        max_array = np.maximum.reduce([model_predictions.values, data])
        residual = ((model_predictions.values - data) / max_array).ravel()

        return residual

    def residual_odeint(
        self,
        parameters,
        t,
        data,
        method="RK45",
        differential_equations_version=1,
        state="New York",
        initial_values=None,
    ):
        residual_odeint = self.residual(
            parameters,
            t,
            data,
            solver="odeint",
            method=method,
            differential_equations_version=differential_equations_version,
            state=state,
            initial_values=initial_values,
        )
        return residual_odeint

    def residual_solve_ivp(
        self,
        parameters,
        t,
        data,
        method="RK45",
        differential_equations_version=1,
        state="New York",
        initial_values=None,
    ):
        residual_solve_ivp = self.residual(
            parameters,
            t,
            data,
            solver="solve_ivp",
            method=method,
            differential_equations_version=differential_equations_version,
            state=state,
            initial_values=initial_values,
        )
        return residual_solve_ivp

    def plot(self, actual_values, model_predictions):
        """This method plots the model predictions vs the actual data.

        :parameter model_predictions: Array - Model predictions."""

        for i, compartment_name in enumerate(self.epidemiological_compartment_names):
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
            plt.show()


epidemiological_model_parameter_computer_configuration = {
    "data_path": f"{DATA_DIR}/epidemiological_model_data/",
    "simulation_start_date": "11/01/2021",
    "epidemiological_compartment_names": [
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
    ],
    "parameter_computation_timeframe": 28,
    "constrained_beta": False,
}

epidemiological_model_parameter_computer = EpidemiologicalModelParameterComputer(
    parameter_computer_configuration=epidemiological_model_parameter_computer_configuration
)

epidemiological_model_parameter_computer.compute_epidemiological_model_parameters()
