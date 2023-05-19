# Imports
import json
import os
from collections import defaultdict
from time import time
import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import minimize, fit_report
from scipy.integrate import odeint, solve_ivp

from src.settings import data_directory
from src.utilities.parameter_initializer import ParameterInitializer


# noinspection DuplicatedCode
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

        self.epidemiological_model_parameters = self.parameter_initializer.initialize_initial_epidemiological_model_parameters(
            self.parameter_computer_configuration["constrained_beta"]
        )

        self.epidemiological_compartment_names = parameter_computer_configuration[
            "epidemiological_compartment_names"
        ]

        self.original_residual = None

    def compute_epidemiological_model_parameters(self, state):
        """This method computes the epidemiological model parameters and saves them..."""

        parameter_computation_timeframe = self.parameter_computer_configuration[
            "parameter_computation_timeframe"
        ]

        # for state in self.epidemiological_model_data:
        # def multi_threaded_parameter_computation(state):
        print("\nState:", state)

        # Starts with a blank file if file exists with previously computed epidemiological model parameters.
        with open(
            f"{data_directory}/epidemiological_model_parameters/goodness_of_fit/{state}.txt",
            "w",
        ) as outfile:
            outfile.close()

        state_runtime_start = time()

        model_predictions = []
        original_residuals = []
        total_residual = 0
        state_parameters = defaultdict(list)

        # We compute the epidemiological model parameters every fixed number of days determined by
        # (parameter_computation_timeframe).
        number_of_splits = int(
            np.ceil(
                self.epidemiological_model_data[state].shape[0]
                / parameter_computation_timeframe
            )
        )

        for split_number in range(number_of_splits):
            print(f"\nSplit {split_number + 1} of {number_of_splits}:")

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

            # Times at which to store the computed solution. This also determines the interval of integration
            # (t0, tf). The solver starts with t=t0 and integrates until it reaches t=tf.
            t = np.linspace(
                0,
                split_epidemiological_data.shape[0] - 1,
                split_epidemiological_data.shape[0],
            )

            # Initial values.
            y0 = [
                split_epidemiological_data[
                    f"{self.epidemiological_compartment_names[i]}"
                ].iloc[0]
                for i in range(len(self.epidemiological_compartment_names))
            ]

            split_runtime_start = time()

            # Note: Args are the additional positional arguments to be passed to self.residual_solve_ivp
            model_fit_solve_ivp = minimize(
                self.residual_solve_ivp,
                self.epidemiological_model_parameters,
                args=(
                    t,
                    epidemiological_model_compartment_values,
                    self.parameter_computer_configuration["integration_method"],
                    self.parameter_computer_configuration[
                        "differential_equation_version"
                    ],
                    state,
                    y0,
                ),
                method=self.parameter_computer_configuration["fitting_method"],
                nan_policy=self.parameter_computer_configuration["nan_policy"],
                max_nfev=self.parameter_computer_configuration[
                    "maximum_number_of_function_evaluations"
                ],
            )

            print("Split Runtime:", time() - split_runtime_start, "seconds")

            model_prediction = (
                epidemiological_model_compartment_values
                + self.original_residual.reshape(
                    epidemiological_model_compartment_values.shape
                )
            )

            model_predictions.append(model_prediction)

            original_residuals.append(self.original_residual)

            for name, parameter in model_fit_solve_ivp.params.items():
                state_parameters[f"{name}"].append(parameter.value)

            # print(
            #     "Split Residual:",
            #     np.sum(np.abs(self.original_residual)),
            # )

            # print(
            #     f"\n\nFit Report {split_number + 1}:\n",
            #     fit_report(model_fit_solve_ivp),
            # )
            # print(f"\n\nParameters {split_number + 1}:")
            # print(model_fit_solve_ivp.params.pretty_print(), "\n")
            # print(
            #     "-----------------------------------------------------------------------------------------------"
            # )

            with open(
                f"{data_directory}/epidemiological_model_parameters/goodness_of_fit/{state}.txt",
                "a",
            ) as outfile:
                outfile.write(
                    f"\n\nSplit {split_number + 1} of {number_of_splits}:"
                )
                outfile.write(
                    f"\nSplit Runtime: {time() - split_runtime_start} seconds"
                )
                outfile.write(
                    f"\nSplit Residual: {np.sum(np.abs(self.original_residual))}"
                )
                outfile.write(
                    f"\n\nFit Report {split_number + 1}:\n {fit_report(model_fit_solve_ivp)}"
                )
                # outfile.write(f"\n\nParameters {split_number + 1}:")
                # outfile.write(f"{model_fit_solve_ivp.params.pretty_print()}, \n")
                outfile.write(
                    "\n--------------------------------------------------------------------------------------------"
                )

        total_residual += np.sum(np.abs(np.asarray(original_residuals[:-1])))

        # print(f"\n\nTotal Residual {state}:", total_residual)
        print(f"{state} Runtime:", time() - state_runtime_start, "seconds\n")

        with open(
            f"{data_directory}/epidemiological_model_parameters/goodness_of_fit/{state}.txt",
            "a",
        ) as outfile:
            outfile.write(f"\n\nTotal Residual {state}: {total_residual}")

        # for parameter in state_parameters.keys():
        #     print(f"{parameter}:", state_parameters[parameter])

        with open(
            f"{data_directory}/epidemiological_model_parameters/{state}.json", "w"
        ) as outfile:
            json.dump(state_parameters, outfile)

        model_predictions = np.concatenate(
            [model_predictions[i] for i in range(len(model_predictions))]
        )

        # Saving the model predictions.
        date_values = self.epidemiological_model_data[state]["date"].values.reshape(
            -1, 1
        )
        data = np.concatenate((date_values, model_predictions), axis=1)
        model_predictions_dataframe = pd.DataFrame(
            data, columns=[["date"] + self.epidemiological_compartment_names]
        )
        model_predictions_dataframe.to_csv(
            f"{data_directory}/epidemiological_model_parameters/model_predictions/{state}.csv",
            index=False,
        )

        self.plot(
            state=state,
            actual_values=self.epidemiological_model_data[state][
                self.epidemiological_compartment_names
            ].values,
            model_predictions=model_predictions,
        )

        # print("CPU Count:", multiprocessing.cpu_count())
        # # if __name__ == "__main__":
        # pool = Pool(4)
        # pool.map(
        #     multi_threaded_parameter_computation, self.epidemiological_model_data
        # )

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
                # det_uv,
                # det_v,
                # det_biv,
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
                # det_5_17_uv,
                # det_5_17_v,
                # det_5_17_biv,
                # det_18_49_uv,
                # det_18_49_v,
                # det_18_49_biv,
                # det_50_64_uv,
                # det_50_64_v,
                # det_50_64_biv,
                # det_65_plus_uv,
                # det_65_plus_v,
                # det_65_plus_biv,
            ) = y

            if i_biv < 0:
                print("Negative I_BiV")
            if i_v < 0:
                print("Negative I_V")
            if i_uv < 0:
                print("Negative I_UV")

            # Force of infection
            total_infections = max((i_uv + i_v + i_biv), 1)

            # Parameter Values
            # beta = parameters["beta"].value
            alpha = parameters["alpha"].value

            # Infection Rates:
            beta_uv = parameters["beta_uv"].value
            beta_v = parameters["beta_v"].value
            beta_biv = parameters["beta_biv"].value
            beta_ruv = parameters["beta_ruv"].value
            beta_rv = parameters["beta_rv"].value
            beta_rbiv = parameters["beta_rbiv"].value

            beta_5_17_uv = parameters["beta_5_17_uv"].value
            beta_5_17_v = parameters["beta_5_17_v"].value
            beta_5_17_biv = parameters["beta_5_17_biv"].value
            beta_5_17_ruv = parameters["beta_5_17_ruv"].value
            beta_5_17_rv = parameters["beta_5_17_rv"].value
            beta_5_17_rbiv = parameters["beta_5_17_rbiv"].value

            beta_18_49_uv = parameters["beta_18_49_uv"].value
            beta_18_49_v = parameters["beta_18_49_v"].value
            beta_18_49_biv = parameters["beta_18_49_biv"].value
            beta_18_49_ruv = parameters["beta_18_49_ruv"].value
            beta_18_49_rv = parameters["beta_18_49_rv"].value
            beta_18_49_rbiv = parameters["beta_18_49_rbiv"].value

            beta_50_64_uv = parameters["beta_50_64_uv"].value
            beta_50_64_v = parameters["beta_50_64_v"].value
            beta_50_64_biv = parameters["beta_50_64_biv"].value
            beta_50_64_ruv = parameters["beta_50_64_ruv"].value
            beta_50_64_rv = parameters["beta_50_64_rv"].value
            beta_50_64_rbiv = parameters["beta_50_64_rbiv"].value

            beta_65_plus_uv = parameters["beta_65_plus_uv"].value
            beta_65_plus_v = parameters["beta_65_plus_v"].value
            beta_65_plus_biv = parameters["beta_65_plus_biv"].value
            beta_65_plus_ruv = parameters["beta_65_plus_ruv"].value
            beta_65_plus_rv = parameters["beta_65_plus_rv"].value
            beta_65_plus_rbiv = parameters["beta_65_plus_rbiv"].value

            # Hospitalization Rates:
            delta_uv = parameters["delta_uv"].value
            delta_v = parameters["delta_v"].value
            delta_biv = parameters["delta_biv"].value

            delta_5_17_uv = parameters["delta_5_17_uv"].value
            delta_5_17_v = parameters["delta_5_17_v"].value
            delta_5_17_biv = parameters["delta_5_17_biv"].value

            delta_18_49_uv = parameters["delta_18_49_uv"].value
            delta_18_49_v = parameters["delta_18_49_v"].value
            delta_18_49_biv = parameters["delta_18_49_biv"].value

            delta_50_64_uv = parameters["delta_50_64_uv"].value
            delta_50_64_v = parameters["delta_50_64_v"].value
            delta_50_64_biv = parameters["delta_50_64_biv"].value

            delta_65_plus_uv = parameters["delta_65_plus_uv"].value
            delta_65_plus_v = parameters["delta_65_plus_v"].value
            delta_65_plus_biv = parameters["delta_65_plus_biv"].value

            # Mortality Rates:
            mu_i_uv = parameters["mu_i_uv"].value
            mu_i_v = parameters["mu_i_v"].value
            mu_i_biv = parameters["mu_i_biv"].value
            mu_h_uv = parameters["mu_h_uv"].value
            mu_h_v = parameters["mu_h_v"].value
            mu_h_biv = parameters["mu_h_biv"].value

            mu_i_5_17_uv = parameters["mu_i_5_17_uv"].value
            mu_i_5_17_v = parameters["mu_i_5_17_v"].value
            mu_i_5_17_biv = parameters["mu_i_5_17_biv"].value
            mu_h_5_17_uv = parameters["mu_h_5_17_uv"].value
            mu_h_5_17_v = parameters["mu_h_5_17_v"].value
            mu_h_5_17_biv = parameters["mu_h_5_17_biv"].value

            mu_i_18_49_uv = parameters["mu_i_18_49_uv"].value
            mu_i_18_49_v = parameters["mu_i_18_49_v"].value
            mu_i_18_49_biv = parameters["mu_i_18_49_biv"].value
            mu_h_18_49_uv = parameters["mu_h_18_49_uv"].value
            mu_h_18_49_v = parameters["mu_h_18_49_v"].value
            mu_h_18_49_biv = parameters["mu_h_18_49_biv"].value

            mu_i_50_64_uv = parameters["mu_i_50_64_uv"].value
            mu_i_50_64_v = parameters["mu_i_50_64_v"].value
            mu_i_50_64_biv = parameters["mu_i_50_64_biv"].value
            mu_h_50_64_uv = parameters["mu_h_50_64_uv"].value
            mu_h_50_64_v = parameters["mu_h_50_64_v"].value
            mu_h_50_64_biv = parameters["mu_h_50_64_biv"].value

            mu_i_65_plus_uv = parameters["mu_i_65_plus_uv"].value
            mu_i_65_plus_v = parameters["mu_i_65_plus_v"].value
            mu_i_65_plus_biv = parameters["mu_i_65_plus_biv"].value
            mu_h_65_plus_uv = parameters["mu_h_65_plus_uv"].value
            mu_h_65_plus_v = parameters["mu_h_65_plus_v"].value
            mu_h_65_plus_biv = parameters["mu_h_65_plus_biv"].value

            # Recovery Rates:
            gamma_i_uv = parameters["gamma_i_uv"].value
            gamma_i_v = parameters["gamma_i_v"].value
            gamma_i_biv = parameters["gamma_i_biv"].value
            gamma_h_uv = parameters["gamma_h_uv"].value
            gamma_h_v = parameters["gamma_h_v"].value
            gamma_h_biv = parameters["gamma_h_biv"].value

            gamma_i_5_17_uv = parameters["gamma_i_5_17_uv"].value
            gamma_i_5_17_v = parameters["gamma_i_5_17_v"].value
            gamma_i_5_17_biv = parameters["gamma_i_5_17_biv"].value
            gamma_h_5_17_uv = parameters["gamma_h_5_17_uv"].value
            gamma_h_5_17_v = parameters["gamma_h_5_17_v"].value
            gamma_h_5_17_biv = parameters["gamma_h_5_17_biv"].value

            gamma_i_18_49_uv = parameters["gamma_i_18_49_uv"].value
            gamma_i_18_49_v = parameters["gamma_i_18_49_v"].value
            gamma_i_18_49_biv = parameters["gamma_i_18_49_biv"].value
            gamma_h_18_49_uv = parameters["gamma_h_18_49_uv"].value
            gamma_h_18_49_v = parameters["gamma_h_18_49_v"].value
            gamma_h_18_49_biv = parameters["gamma_h_18_49_biv"].value

            gamma_i_50_64_uv = parameters["gamma_i_50_64_uv"].value
            gamma_i_50_64_v = parameters["gamma_i_50_64_v"].value
            gamma_i_50_64_biv = parameters["gamma_i_50_64_biv"].value
            gamma_h_50_64_uv = parameters["gamma_h_50_64_uv"].value
            gamma_h_50_64_v = parameters["gamma_h_50_64_v"].value
            gamma_h_50_64_biv = parameters["gamma_h_50_64_biv"].value

            gamma_i_65_plus_uv = parameters["gamma_i_65_plus_uv"].value
            gamma_i_65_plus_v = parameters["gamma_i_65_plus_v"].value
            gamma_i_65_plus_biv = parameters["gamma_i_65_plus_biv"].value
            gamma_h_65_plus_uv = parameters["gamma_h_65_plus_uv"].value
            gamma_h_65_plus_v = parameters["gamma_h_65_plus_v"].value
            gamma_h_65_plus_biv = parameters["gamma_h_65_plus_biv"].value

            # Ordinary Differential Equations.
            # beta = beta_s * self.epidemiological_model_data[state]['New Cases'].iloc[
            #     min(int(t), len(self.epidemiological_model_data[state]) - 1)]
            # beta = beta_s

            # Susceptible
            ds_uv_dt = (
                -beta_uv * s_uv * (total_infections**alpha) / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_uv
            )
            ds_v_dt = (
                -beta_v * s_v * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_v
            )
            ds_biv_dt = (
                -beta_biv * s_biv * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_v
            )

            ds_5_17_uv_dt = (
                -beta_5_17_uv * s_5_17_uv * (total_infections**alpha) / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_5_17_uv
            )
            ds_5_17_v_dt = (
                -beta_5_17_v * s_5_17_v * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_5_17_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_5_17_v
            )
            ds_5_17_biv_dt = (
                -beta_5_17_biv * s_5_17_biv * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_5_17_v
            )

            ds_18_49_uv_dt = (
                -beta_18_49_uv * s_18_49_uv * (total_infections**alpha) / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_18_49_uv
            )
            ds_18_49_v_dt = (
                -beta_18_49_v * s_18_49_v * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_18_49_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_18_49_v
            )
            ds_18_49_biv_dt = (
                -beta_18_49_biv * s_18_49_biv * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_18_49_v
            )

            ds_50_64_uv_dt = (
                -beta_50_64_uv * s_50_64_uv * (total_infections**alpha) / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_50_64_uv
            )
            ds_50_64_v_dt = (
                -beta_50_64_v * s_50_64_v * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_50_64_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_50_64_v
            )
            ds_50_64_biv_dt = (
                -beta_50_64_biv * s_50_64_biv * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_50_64_v
            )

            ds_65_plus_uv_dt = (
                -beta_65_plus_uv
                * s_65_plus_uv
                * (total_infections**alpha)
                / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_65_plus_uv
            )
            ds_65_plus_v_dt = (
                -beta_65_plus_v * s_65_plus_v * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_65_plus_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_65_plus_v
            )
            ds_65_plus_biv_dt = (
                -beta_65_plus_biv
                * s_65_plus_biv
                * (total_infections**alpha)
                / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * s_65_plus_v
            )

            # Infected
            di_uv_dt = (
                beta_uv * s_uv * (total_infections**alpha) / population
                + beta_ruv * r_uv * (total_infections**alpha) / population
                - delta_uv * i_uv
                - gamma_i_uv * i_uv
                - mu_i_uv * i_uv
            )
            di_v_dt = (
                beta_v * s_v * (total_infections**alpha) / population
                + beta_rv * r_v * (total_infections**alpha) / population
                - delta_v * i_v
                - gamma_i_v * i_v
                - mu_i_v * i_v
            )
            di_biv_dt = (
                beta_biv * s_biv * (total_infections**alpha) / population
                + beta_rbiv * r_biv * (total_infections**alpha) / population
                - delta_biv * i_biv
                - gamma_i_biv * i_biv
                - mu_i_biv * i_biv
            )

            di_5_17_uv_dt = (
                beta_5_17_uv * s_5_17_uv * (total_infections**alpha) / population
                + beta_5_17_ruv * r_5_17_uv * (total_infections**alpha) / population
                - delta_5_17_uv * i_5_17_uv
                - gamma_i_5_17_uv * i_5_17_uv
                - mu_i_5_17_uv * i_5_17_uv
            )
            di_5_17_v_dt = (
                beta_5_17_v * s_5_17_v * (total_infections**alpha) / population
                + beta_5_17_rv * r_5_17_v * (total_infections**alpha) / population
                - delta_5_17_v * i_5_17_v
                - gamma_i_5_17_v * i_5_17_v
                - mu_i_5_17_v * i_5_17_v
            )
            di_5_17_biv_dt = (
                beta_5_17_biv * s_5_17_biv * (total_infections**alpha) / population
                + beta_5_17_rbiv * r_5_17_biv * (total_infections**alpha) / population
                - delta_5_17_biv * i_5_17_biv
                - gamma_i_5_17_biv * i_5_17_biv
                - mu_i_5_17_biv * i_5_17_biv
            )

            di_18_49_uv_dt = (
                beta_18_49_uv * s_18_49_uv * (total_infections**alpha) / population
                + beta_18_49_ruv * r_18_49_uv * (total_infections**alpha) / population
                - delta_18_49_uv * i_18_49_uv
                - gamma_i_18_49_uv * i_18_49_uv
                - mu_i_18_49_uv * i_18_49_uv
            )
            di_18_49_v_dt = (
                beta_18_49_v * s_18_49_v * (total_infections**alpha) / population
                + beta_18_49_rv * r_18_49_v * (total_infections**alpha) / population
                - delta_18_49_v * i_18_49_v
                - gamma_i_18_49_v * i_18_49_v
                - mu_i_18_49_v * i_18_49_v
            )
            di_18_49_biv_dt = (
                beta_18_49_biv * s_18_49_biv * (total_infections**alpha) / population
                + beta_18_49_rbiv
                * r_18_49_biv
                * (total_infections**alpha)
                / population
                - delta_18_49_biv * i_18_49_biv
                - gamma_i_18_49_biv * i_18_49_biv
                - mu_i_18_49_biv * i_18_49_biv
            )

            di_50_64_uv_dt = (
                beta_50_64_uv * s_50_64_uv * (total_infections**alpha) / population
                + beta_50_64_ruv * r_50_64_uv * (total_infections**alpha) / population
                - delta_50_64_uv * i_50_64_uv
                - gamma_i_50_64_uv * i_50_64_uv
                - mu_i_50_64_uv * i_50_64_uv
            )
            di_50_64_v_dt = (
                beta_50_64_v * s_50_64_v * (total_infections**alpha) / population
                + beta_50_64_rv * r_50_64_v * (total_infections**alpha) / population
                - delta_50_64_v * i_50_64_v
                - gamma_i_50_64_v * i_50_64_v
                - mu_i_50_64_v * i_50_64_v
            )
            di_50_64_biv_dt = (
                beta_50_64_biv * s_50_64_biv * (total_infections**alpha) / population
                + beta_50_64_rbiv
                * r_50_64_biv
                * (total_infections**alpha)
                / population
                - delta_50_64_biv * i_50_64_biv
                - gamma_i_50_64_biv * i_50_64_biv
                - mu_i_50_64_biv * i_50_64_biv
            )

            di_65_plus_uv_dt = (
                beta_65_plus_uv
                * s_65_plus_uv
                * (total_infections**alpha)
                / population
                + beta_65_plus_ruv
                * r_65_plus_uv
                * (total_infections**alpha)
                / population
                - delta_65_plus_uv * i_65_plus_uv
                - gamma_i_65_plus_uv * i_65_plus_uv
                - mu_i_65_plus_uv * i_65_plus_uv
            )
            di_65_plus_v_dt = (
                beta_65_plus_v * s_65_plus_v * (total_infections**alpha) / population
                + beta_65_plus_rv
                * r_65_plus_v
                * (total_infections**alpha)
                / population
                - delta_65_plus_v * i_65_plus_v
                - gamma_i_65_plus_v * i_65_plus_v
                - mu_i_65_plus_v * i_65_plus_v
            )
            di_65_plus_biv_dt = (
                beta_65_plus_biv
                * s_65_plus_biv
                * (total_infections**alpha)
                / population
                + beta_65_plus_rbiv
                * r_65_plus_biv
                * (total_infections**alpha)
                / population
                - delta_65_plus_biv * i_65_plus_biv
                - gamma_i_65_plus_biv * i_65_plus_biv
                - mu_i_65_plus_biv * i_65_plus_biv
            )

            # Hospitalized
            dh_uv_dt = delta_uv * i_uv - gamma_h_uv * h_uv - mu_h_uv * h_uv
            dh_v_dt = delta_v * i_v - gamma_h_v * h_v - mu_h_v * h_v
            dh_biv_dt = delta_biv * i_biv - gamma_h_biv * h_biv - mu_h_biv * h_biv

            dh_5_17_uv_dt = (
                delta_5_17_uv * i_5_17_uv
                - gamma_h_5_17_uv * h_5_17_uv
                - mu_h_5_17_uv * h_5_17_uv
            )
            dh_5_17_v_dt = (
                delta_5_17_v * i_5_17_v
                - gamma_h_5_17_v * h_5_17_v
                - mu_h_5_17_v * h_5_17_v
            )
            dh_5_17_biv_dt = (
                delta_5_17_biv * i_5_17_biv
                - gamma_h_5_17_biv * h_5_17_biv
                - mu_h_5_17_biv * h_5_17_biv
            )

            dh_18_49_uv_dt = (
                delta_18_49_uv * i_18_49_uv
                - gamma_h_18_49_uv * h_18_49_uv
                - mu_h_18_49_uv * h_18_49_uv
            )
            dh_18_49_v_dt = (
                delta_18_49_v * i_18_49_v
                - gamma_h_18_49_v * h_18_49_v
                - mu_h_18_49_v * h_18_49_v
            )
            dh_18_49_biv_dt = (
                delta_18_49_biv * i_18_49_biv
                - gamma_h_18_49_biv * h_18_49_biv
                - mu_h_18_49_biv * h_18_49_biv
            )

            dh_50_64_uv_dt = (
                delta_50_64_uv * i_50_64_uv
                - gamma_h_50_64_uv * h_50_64_uv
                - mu_h_50_64_uv * h_50_64_uv
            )
            dh_50_64_v_dt = (
                delta_50_64_v * i_50_64_v
                - gamma_h_50_64_v * h_50_64_v
                - mu_h_50_64_v * h_50_64_v
            )
            dh_50_64_biv_dt = (
                delta_50_64_biv * i_50_64_biv
                - gamma_h_50_64_biv * h_50_64_biv
                - mu_h_50_64_biv * h_50_64_biv
            )

            dh_65_plus_uv_dt = (
                delta_65_plus_uv * i_65_plus_uv
                - gamma_h_65_plus_uv * h_65_plus_uv
                - mu_h_65_plus_uv * h_65_plus_uv
            )
            dh_65_plus_v_dt = (
                delta_65_plus_v * i_65_plus_v
                - gamma_h_65_plus_v * h_65_plus_v
                - mu_h_65_plus_v * h_65_plus_v
            )
            dh_65_plus_biv_dt = (
                delta_65_plus_biv * i_65_plus_biv
                - gamma_h_65_plus_biv * h_65_plus_biv
                - mu_h_65_plus_biv * h_65_plus_biv
            )

            # Recovered
            dr_uv_dt = (
                gamma_i_uv * i_uv
                + gamma_h_uv * h_uv
                - beta_ruv * r_uv * (total_infections**alpha) / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_uv
            )
            dr_v_dt = (
                gamma_i_v * i_v
                + gamma_h_v * h_v
                - beta_rv * r_v * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_v
            )
            dr_biv_dt = (
                gamma_i_biv * i_biv
                + gamma_h_biv * h_biv
                - beta_rbiv * r_biv * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_v
            )

            dr_5_17_uv_dt = (
                gamma_i_5_17_uv * i_5_17_uv
                + gamma_h_5_17_uv * h_5_17_uv
                - beta_5_17_ruv * r_5_17_uv * (total_infections**alpha) / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_5_17_uv
            )
            dr_5_17_v_dt = (
                gamma_i_5_17_v * i_5_17_v
                + gamma_h_5_17_v * h_5_17_v
                - beta_5_17_rv * r_5_17_v * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_5_17_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_5_17_v
            )
            dr_5_17_biv_dt = (
                gamma_i_5_17_biv * i_5_17_biv
                + gamma_h_5_17_biv * h_5_17_biv
                - beta_5_17_rbiv * r_5_17_biv * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_5_17_v
            )

            dr_18_49_uv_dt = (
                gamma_i_18_49_uv * i_18_49_uv
                + gamma_h_18_49_uv * h_18_49_uv
                - beta_18_49_ruv * r_18_49_uv * (total_infections**alpha) / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_18_49_uv
            )
            dr_18_49_v_dt = (
                gamma_i_18_49_v * i_18_49_v
                + gamma_h_18_49_v * h_18_49_v
                - beta_18_49_rv * r_18_49_v * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_18_49_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_18_49_v
            )
            dr_18_49_biv_dt = (
                gamma_i_18_49_biv * i_18_49_biv
                + gamma_h_18_49_biv * h_18_49_biv
                - beta_18_49_rbiv
                * r_18_49_biv
                * (total_infections**alpha)
                / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_18_49_v
            )

            dr_50_64_uv_dt = (
                gamma_i_50_64_uv * i_50_64_uv
                + gamma_h_50_64_uv * h_50_64_uv
                - beta_50_64_ruv * r_50_64_uv * (total_infections**alpha) / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_50_64_uv
            )
            dr_50_64_v_dt = (
                gamma_i_50_64_v * i_50_64_v
                + gamma_h_50_64_v * h_50_64_v
                - beta_50_64_rv * r_50_64_v * (total_infections**alpha) / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_50_64_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_50_64_v
            )
            dr_50_64_biv_dt = (
                gamma_i_50_64_biv * i_50_64_biv
                + gamma_h_50_64_biv * h_50_64_biv
                - beta_50_64_rbiv
                * r_50_64_biv
                * (total_infections**alpha)
                / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_50_64_v
            )

            dr_65_plus_uv_dt = (
                gamma_i_65_plus_uv * i_65_plus_uv
                + gamma_h_65_plus_uv * h_65_plus_uv
                - beta_65_plus_ruv
                * r_65_plus_uv
                * (total_infections**alpha)
                / population
                - self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_65_plus_uv
            )
            dr_65_plus_v_dt = (
                gamma_i_65_plus_v * i_65_plus_v
                + gamma_h_65_plus_v * h_65_plus_v
                - beta_65_plus_rv
                * r_65_plus_v
                * (total_infections**alpha)
                / population
                + self.epidemiological_model_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_65_plus_uv
                - self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_65_plus_v
            )
            dr_65_plus_biv_dt = (
                gamma_i_65_plus_biv * i_65_plus_biv
                + gamma_h_65_plus_biv * h_65_plus_biv
                - beta_65_plus_rbiv
                * r_65_plus_biv
                * (total_infections**alpha)
                / population
                + self.epidemiological_model_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[min(int(t), len(self.epidemiological_model_data[state]) - 1)]
                * r_65_plus_v
            )
            # Deceased
            ddec_uv_dt = mu_i_uv * i_uv + mu_h_uv * h_uv
            ddec_v_dt = mu_i_v * i_v + mu_h_v * h_v
            ddec_biv_dt = mu_i_biv * i_biv + mu_h_biv * h_biv

            ddec_5_17_uv_dt = mu_i_5_17_uv * i_5_17_uv + mu_h_5_17_uv * h_5_17_uv
            ddec_5_17_v_dt = mu_i_5_17_v * i_5_17_v + mu_h_5_17_v * h_5_17_v
            ddec_5_17_biv_dt = mu_i_5_17_biv * i_5_17_biv + mu_h_5_17_biv * h_5_17_biv

            ddec_18_49_uv_dt = mu_i_18_49_uv * i_18_49_uv + mu_h_18_49_uv * h_18_49_uv
            ddec_18_49_v_dt = mu_i_18_49_v * i_18_49_v + mu_h_18_49_v * h_18_49_v
            ddec_18_49_biv_dt = (
                mu_i_18_49_biv * i_18_49_biv + mu_h_18_49_biv * h_18_49_biv
            )

            ddec_50_64_uv_dt = mu_i_50_64_uv * i_50_64_uv + mu_h_50_64_uv * h_50_64_uv
            ddec_50_64_v_dt = mu_i_50_64_v * i_50_64_v + mu_h_50_64_v * h_50_64_v
            ddec_50_64_biv_dt = (
                mu_i_50_64_biv * i_50_64_biv + mu_h_50_64_biv * h_50_64_biv
            )

            ddec_65_plus_uv_dt = (
                mu_i_65_plus_uv * i_65_plus_uv + mu_h_65_plus_uv * h_65_plus_uv
            )
            ddec_65_plus_v_dt = (
                mu_i_65_plus_v * i_65_plus_v + mu_h_65_plus_v * h_65_plus_v
            )
            ddec_65_plus_biv_dt = (
                mu_i_65_plus_biv * i_65_plus_biv + mu_h_65_plus_biv * h_65_plus_biv
            )

            return (
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
                # ddet_5_17_uv_dt,
                # ddet_5_17_v_dt,
                # ddet_5_17_biv_dt,
                # ddet_18_49_uv_dt,
                # ddet_18_49_v_dt,
                # ddet_18_49_biv_dt,
                # ddet_50_64_uv_dt,
                # ddet_50_64_v_dt,
                # ddet_50_64_biv_dt,
                # ddet_65_plus_uv_dt,
                # ddet_65_plus_v_dt,
                # ddet_65_plus_biv_dt,
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
                func=self.differential_equations,
                y0=y0,
                t=t,
                args=(
                    population,
                    parameters,
                    True,
                    differential_equations_version,
                    state,
                ),
            )

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

    def plot(self, state, actual_values, model_predictions):
        """This method plots the model predictions vs the actual data.

        :parameter model_predictions: Array - Model predictions."""

        for i, compartment_name in enumerate(self.epidemiological_compartment_names):
            if not os.path.exists(
                f"{data_directory}/epidemiological_model_parameters/plots/{state}"
            ):
                os.makedirs(
                    f"{data_directory}/epidemiological_model_parameters/plots/{state}"
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
            plt.savefig(
                f"{data_directory}/epidemiological_model_parameters/plots/{state}/{compartment_name}.png"
            )
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
        # "Detected",
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

    epidemiological_model_parameter_computer_configuration = {
        "data_path": f"{data_directory}/epidemiological_model_data/",
        "output_path": f"{data_directory}/epidemiological_model_parameters/",
        "simulation_start_date": "11/01/2021",
        "epidemiological_compartment_names": epidemiological_compartments,
        "parameter_computation_timeframe": 28,
        "constrained_beta": False,
        "integration_method": "RK45",
        "differential_equation_version": 1,
        "fitting_method": "leastsq",
        "nan_policy": "omit",
        "maximum_number_of_function_evaluations": 10_000,
    }

    epidemiological_model_parameter_computer = EpidemiologicalModelParameterComputer(
        parameter_computer_configuration=epidemiological_model_parameter_computer_configuration
    )
    # epidemiological_model_parameter_computer.compute_epidemiological_model_parameters()

    print("CPU Count:", multiprocessing.cpu_count())
    pool = Pool(24)
    pool.map(epidemiological_model_parameter_computer.compute_epidemiological_model_parameters,
             epidemiological_model_parameter_computer.epidemiological_model_data)
