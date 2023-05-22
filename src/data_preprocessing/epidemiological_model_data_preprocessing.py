import pandas as pd

from src.settings import data_directory
from src.utilities.parameter_initializer import ParameterInitializer

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None


class EpidemiologicalDataPreProcessing:
    """This class creates the epidemiological data to be used for computing the parameters of the epidemiological
    model."""

    def __init__(self, data_paths):
        """This method loads the data for pre-processing.

        :param data_paths:"""

        self.data_paths = data_paths
        self.parameter_initializer = ParameterInitializer(
            data_paths["processed_state_data"]
        )

        self.states = self.parameter_initializer.initialize_state_names()

        self.epidemiological_data = (
            self.parameter_initializer.initialize_epidemiological_model_data()
        )

        # TODO: Look into how significant the difference between state populations is at various points in time in your
        #       data.
        self.state_populations = (
            self.parameter_initializer.initialize_state_populations()
        )

        self.cases_by_age_vaccination = pd.read_csv(
            data_paths["cases_by_age_vaccination"]
        )
        self.deaths_by_age_vaccination = pd.read_csv(
            data_paths["deaths_by_age_vaccination"]
        )
        self.hospitalizations_by_age_vaccination = (
            pd.read_csv(data_paths["hospitalizations_by_age_vaccination"])
            .iloc[386:1113]
            .reset_index(drop=True)
        )

    def compute_age_group_and_vaccination_status_multipliers(self):
        """This method computes the age group and vaccination status wise multipliers for splitting the data."""

        age_groups = ["5-17", "18-49", "50-64", "65+"]
        vaccination_groups = ["UV", "V", "BiV"]
        datasets = [self.cases_by_age_vaccination, self.deaths_by_age_vaccination]

        # Here we compute the total population and incidence rate per age group.
        for dataset in datasets:
            for age_group in age_groups:
                dataset[
                    [
                        f"{age_group}_IR",
                        f"{age_group}_Population",
                        f"{age_group}_Multiplier",
                    ]
                ] = 0
                for i in range(len(dataset)):
                    age_group_population = (
                        dataset[f"{age_group}_UV_Population"].iloc[i]
                        + dataset[f"{age_group}_V_Population"].iloc[i]
                        + dataset[f"{age_group}_BiV_Population"].iloc[i]
                    )

                    dataset[f"{age_group}_Population"].iloc[i] = age_group_population

                    dataset[f"{age_group}_IR"].iloc[i] = (
                        dataset[f"{age_group}_UV_IR"].iloc[i]
                        * dataset[f"{age_group}_UV_Population"].iloc[i]
                        + dataset[f"{age_group}_V_IR"].iloc[i]
                        * dataset[f"{age_group}_V_Population"].iloc[i]
                        + dataset[f"{age_group}_BiV_IR"].iloc[i]
                        * dataset[f"{age_group}_BiV_Population"].iloc[i]
                    ) / age_group_population

            # Here we compute the total population and incidence rate per vaccination group.
            for vaccination_group in vaccination_groups:
                dataset[
                    [
                        f"{vaccination_group}_IR",
                        f"{vaccination_group}_Population",
                        f"{vaccination_group}_Multiplier",
                    ]
                ] = 0
                for i in range(len(dataset)):
                    vaccination_group_population = (
                        dataset[f"5-17_{vaccination_group}_Population"].iloc[i]
                        + dataset[f"18-49_{vaccination_group}_Population"].iloc[i]
                        + dataset[f"50-64_{vaccination_group}_Population"].iloc[i]
                        + dataset[f"65+_{vaccination_group}_Population"].iloc[i]
                    )

                    dataset[f"{vaccination_group}_Population"].iloc[
                        i
                    ] = vaccination_group_population

                    dataset[f"{vaccination_group}_IR"].iloc[i] = (
                        0
                        if vaccination_group_population == 0
                        else (
                            dataset[f"5-17_{vaccination_group}_IR"].iloc[i]
                            * dataset[f"5-17_{vaccination_group}_Population"].iloc[i]
                            + dataset[f"18-49_{vaccination_group}_IR"].iloc[i]
                            * dataset[f"18-49_{vaccination_group}_Population"].iloc[i]
                            + dataset[f"50-64_{vaccination_group}_IR"].iloc[i]
                            * dataset[f"50-64_{vaccination_group}_Population"].iloc[i]
                            + dataset[f"65+_{vaccination_group}_IR"].iloc[i]
                            * dataset[f"65+_{vaccination_group}_Population"].iloc[i]
                        )
                        / vaccination_group_population
                    )

            # Multipliers
            for i in range(len(dataset)):
                uv = dataset["UV_IR"].iloc[i] * dataset["UV_Population"].iloc[i]
                v = dataset["V_IR"].iloc[i] * dataset["V_Population"].iloc[i]
                biv = dataset["BiV_IR"].iloc[i] * dataset["BiV_Population"].iloc[i]

                dataset["UV_Multiplier"].iloc[i] = uv / (uv + v + biv)
                dataset["V_Multiplier"].iloc[i] = v / (uv + v + biv)
                dataset["BiV_Multiplier"].iloc[i] = biv / (uv + v + biv)

                _5_17 = dataset["5-17_IR"].iloc[i] * dataset["5-17_Population"].iloc[i]
                _18_49 = (
                    dataset["18-49_IR"].iloc[i] * dataset["18-49_Population"].iloc[i]
                )
                _50_64 = (
                    dataset["50-64_IR"].iloc[i] * dataset["50-64_Population"].iloc[i]
                )
                _65_plus = dataset["65+_IR"].iloc[i] * dataset["65+_Population"].iloc[i]

                dataset["5-17_Multiplier"].iloc[i] = _5_17 / (
                    _5_17 + _18_49 + _50_64 + _65_plus
                )
                dataset["18-49_Multiplier"].iloc[i] = _18_49 / (
                    _5_17 + _18_49 + _50_64 + _65_plus
                )
                dataset["50-64_Multiplier"].iloc[i] = _50_64 / (
                    _5_17 + _18_49 + _50_64 + _65_plus
                )
                dataset["65+_Multiplier"].iloc[i] = _65_plus / (
                    _5_17 + _18_49 + _50_64 + _65_plus
                )

            for age_group in age_groups:
                dataset[
                    [
                        f"{age_group}_UV_Multiplier",
                        f"{age_group}_V_Multiplier",
                        f"{age_group}_BiV_Multiplier",
                    ]
                ] = 0
                for i in range(len(dataset)):
                    uv = (
                        dataset[f"{age_group}_UV_IR"].iloc[i]
                        * dataset[f"{age_group}_UV_Population"].iloc[i]
                    )
                    v = (
                        dataset[f"{age_group}_V_IR"].iloc[i]
                        * dataset[f"{age_group}_V_Population"].iloc[i]
                    )
                    biv = (
                        dataset[f"{age_group}_BiV_IR"].iloc[i]
                        * dataset[f"{age_group}_BiV_Population"].iloc[i]
                    )

                    dataset[f"{age_group}_UV_Multiplier"].iloc[i] = (
                        0 if (uv + v + biv) == 0 else uv / (uv + v + biv)
                    )
                    dataset[f"{age_group}_V_Multiplier"].iloc[i] = (
                        0 if (uv + v + biv) == 0 else v / (uv + v + biv)
                    )
                    dataset[f"{age_group}_BiV_Multiplier"].iloc[i] = (
                        0 if (uv + v + biv) == 0 else biv / (uv + v + biv)
                    )

        # print(
        #     "Cases by Age and Vaccination First:\n",
        #     self.cases_by_age_vaccination.head(),
        # )
        # print(
        #     "\n\nCases by Age and Vaccination Last:\n",
        #     self.cases_by_age_vaccination.iloc[-5:],
        # )
        #
        # print(
        #     "\n\nDeaths by Age and Vaccination First:\n",
        #     self.deaths_by_age_vaccination.head(),
        # )
        # print(
        #     "\n\nDeaths by Age and Vaccination Last:\n",
        #     self.deaths_by_age_vaccination.iloc[-5:],
        # )

        # Hospitalization Data
        self.hospitalizations_by_age_vaccination[
            ["UV_Multiplier", "V_Multiplier", "BiV_Multiplier"]
        ] = 0

        for i in range(len(self.hospitalizations_by_age_vaccination)):
            uv = (
                self.hospitalizations_by_age_vaccination["UV_IR"].iloc[i]
                * self.cases_by_age_vaccination["UV_Population"].iloc[i]
            )
            v = (
                self.hospitalizations_by_age_vaccination["V_IR"].iloc[i]
                * self.cases_by_age_vaccination["V_Population"].iloc[i]
            )
            biv = (
                self.hospitalizations_by_age_vaccination["BiV_IR"].iloc[i]
                * self.cases_by_age_vaccination["BiV_Population"].iloc[i]
            )

            self.hospitalizations_by_age_vaccination["UV_Multiplier"].iloc[i] = uv / (
                uv + v + biv
            )
            self.hospitalizations_by_age_vaccination["V_Multiplier"].iloc[i] = v / (
                uv + v + biv
            )
            self.hospitalizations_by_age_vaccination["BiV_Multiplier"].iloc[i] = biv / (
                uv + v + biv
            )

        for age_group in age_groups:
            self.hospitalizations_by_age_vaccination[
                [
                    f"{age_group}_UV_Multiplier",
                    f"{age_group}_V_Multiplier",
                    f"{age_group}_BiV_Multiplier",
                ]
            ] = 0

            for i in range(len(self.hospitalizations_by_age_vaccination)):
                uv = (
                    self.hospitalizations_by_age_vaccination[f"{age_group} YR"].iloc[i]
                    / 100
                ) * self.cases_by_age_vaccination["UV_Multiplier"].iloc[i]
                v = (
                    self.hospitalizations_by_age_vaccination[f"{age_group} YR"].iloc[i]
                    / 100
                ) * self.cases_by_age_vaccination["V_Multiplier"].iloc[i]
                biv = (
                    self.hospitalizations_by_age_vaccination[f"{age_group} YR"].iloc[i]
                    / 100
                ) * self.cases_by_age_vaccination["BiV_Multiplier"].iloc[i]

                self.hospitalizations_by_age_vaccination[
                    f"{age_group}_UV_Multiplier"
                ].iloc[i] = uv
                self.hospitalizations_by_age_vaccination[
                    f"{age_group}_V_Multiplier"
                ].iloc[i] = v
                self.hospitalizations_by_age_vaccination[
                    f"{age_group}_BiV_Multiplier"
                ].iloc[i] = biv

        # print(
        #     "\n\nHospitalizations by Age and Vaccination First:\n",
        #     self.hospitalizations_by_age_vaccination.head(),
        # )
        # print(
        #     "\n\nHospitalizations by Age and Vaccination Last:\n",
        #     self.hospitalizations_by_age_vaccination.iloc[-5:],
        # )

    def data_preprocessing(self):
        """This method pre-processes the data for the sub-compartments in the epidemiological model."""

        age_groups = ["5-17", "18-49", "50-64", "65+"]
        vaccination_groups = ["UV", "V", "BiV"]

        for state in self.states:
            self.epidemiological_data[state]["date"] = pd.to_datetime(
                self.epidemiological_data[state]["date"]
            )

            self.epidemiological_data[state] = self.epidemiological_data[state].iloc[
                111:838
            ]
            self.epidemiological_data[state].reset_index(inplace=True, drop=True)
            # print(self.epidemiological_data[state].head())
            # print(self.epidemiological_data[state].iloc[-5:])

            # Vaccination compartments.
            # self.epidemiological_data[state]["unvaccinated_individuals"] = (
            #     self.state_populations[state]
            #     - self.epidemiological_data[state]["Series_Complete_Yes"]
            # )

            self.epidemiological_data[state][
                "primary_series_vaccinated_individuals"
            ] = (
                self.epidemiological_data[state]["Series_Complete_Yes"]
                - self.epidemiological_data[state]["Additional_Doses"]
                # - self.epidemiological_data[state]["Second_Booster"]
                # - self.epidemiological_data[state]["Administered_Bivalent"]
            )

            self.epidemiological_data[state]["first_booster_vaccinated_individuals"] = (
                self.epidemiological_data[state]["Additional_Doses"]
                - self.epidemiological_data[state]["Second_Booster_50Plus"]
                - self.epidemiological_data[state]["Administered_Bivalent"]
            )

            self.epidemiological_data[state][
                "second_booster_vaccinated_individuals"
            ] = (
                self.epidemiological_data[state]["Second_Booster_50Plus"]
                # - self.epidemiological_data[state]["Administered_Bivalent"]
            )

            self.epidemiological_data[state]["vaccinated_individuals"] = (
                self.epidemiological_data[state][
                    "primary_series_vaccinated_individuals"
                ]
                + self.epidemiological_data[state][
                    "first_booster_vaccinated_individuals"
                ]
                + self.epidemiological_data[state][
                    "second_booster_vaccinated_individuals"
                ]
            )

            self.epidemiological_data[state][
                "bivalent_booster_vaccinated_individuals"
            ] = self.epidemiological_data[state]["Administered_Bivalent"]

            self.epidemiological_data[state]["unvaccinated_individuals"] = (
                self.state_populations[state]
                - self.epidemiological_data[state][
                    "primary_series_vaccinated_individuals"
                ]
                - self.epidemiological_data[state][
                    "first_booster_vaccinated_individuals"
                ]
                - self.epidemiological_data[state][
                    "second_booster_vaccinated_individuals"
                ]
                - self.epidemiological_data[state][
                    "bivalent_booster_vaccinated_individuals"
                ]
            )

            # Computing the vaccination rates.
            self.epidemiological_data[state][
                [
                    "percentage_unvaccinated_to_vaccinated",
                    "percentage_vaccinated_to_bivalent_vaccinated",
                ]
            ] = 0

            for i in range(1, len(self.epidemiological_data[state])):
                # Unvaccinated to Fully Vaccinated.
                self.epidemiological_data[state][
                    "percentage_unvaccinated_to_vaccinated"
                ].iloc[i] = (
                    self.epidemiological_data[state]["unvaccinated_individuals"].iloc[
                        i - 1
                    ]
                    - self.epidemiological_data[state]["unvaccinated_individuals"].iloc[
                        i
                    ]
                ) / self.epidemiological_data[
                    state
                ][
                    "unvaccinated_individuals"
                ].iloc[
                    i - 1
                ]

                # Fully Vaccinated to Bivalent Vaccinated.
                self.epidemiological_data[state][
                    "percentage_vaccinated_to_bivalent_vaccinated"
                ].iloc[i] = (
                    self.epidemiological_data[state][
                        "bivalent_booster_vaccinated_individuals"
                    ].iloc[i]
                    - self.epidemiological_data[state][
                        "bivalent_booster_vaccinated_individuals"
                    ].iloc[i - 1]
                ) / self.epidemiological_data[
                    state
                ][
                    "vaccinated_individuals"
                ].iloc[
                    i - 1
                ]

            # # Exposed compartments.
            # exposure_multiplier = (
            #     100 / 0.7
            # )  # We have a reference for this. (Cited > 700 times).
            # self.epidemiological_data[state]["Exposed"] = (
            #     self.epidemiological_data[state]["Daily Cases"] * exposure_multiplier
            # ).astype(int)

            # Susceptible compartments.
            self.epidemiological_data[state]["Susceptible"] = (
                self.state_populations[state]
                # - self.epidemiological_data[state]["Exposed"]
                - self.epidemiological_data[state]["Active Cases"]
                - self.epidemiological_data[state]["Total Recovered"]
                - self.epidemiological_data[state]["Total Deaths (Linear)"]
            )

            # Infected compartments.
            cdc_skew = (
                (
                    self.epidemiological_data[state]["Active Cases"]
                    - self.epidemiological_data[state]["inpatient_beds_used_covid"]
                )
                * (self.cases_by_age_vaccination["UV_Multiplier"])
            ).astype(int)
            self.epidemiological_data[state]["Infected_UV"] = cdc_skew

            cdc_skew = (
                (
                    self.epidemiological_data[state]["Active Cases"]
                    - self.epidemiological_data[state]["inpatient_beds_used_covid"]
                )
                * (self.cases_by_age_vaccination["V_Multiplier"])
            ).astype(int)
            self.epidemiological_data[state]["Infected_V"] = cdc_skew

            cdc_skew = (
                (
                    self.epidemiological_data[state]["Active Cases"]
                    - self.epidemiological_data[state]["inpatient_beds_used_covid"]
                )
                * (self.cases_by_age_vaccination["BiV_Multiplier"])
            ).astype(int)
            self.epidemiological_data[state]["Infected_BiV"] = cdc_skew

            for vaccination_group in vaccination_groups:
                for age_group in age_groups:
                    self.epidemiological_data[state][
                        f"Infected_{age_group}_{vaccination_group}"
                    ] = (
                        (
                            self.epidemiological_data[state]["Active Cases"]
                            - self.epidemiological_data[state][
                                "inpatient_beds_used_covid"
                            ]
                        )
                        * (self.cases_by_age_vaccination[f"{age_group}_Multiplier"])
                        * self.cases_by_age_vaccination[
                            f"{age_group}_{vaccination_group}_Multiplier"
                        ]
                    ).astype(
                        int
                    )

            # Hospitalized compartments.
            cdc_skew = (
                self.epidemiological_data[state]["inpatient_beds_used_covid"]
                * self.hospitalizations_by_age_vaccination["UV_Multiplier"]
            ).astype(int)
            self.epidemiological_data[state]["Hospitalized_UV"] = cdc_skew

            cdc_skew = (
                self.epidemiological_data[state]["inpatient_beds_used_covid"]
                * self.hospitalizations_by_age_vaccination["V_Multiplier"]
            ).astype(int)
            self.epidemiological_data[state]["Hospitalized_V"] = cdc_skew

            cdc_skew = (
                self.epidemiological_data[state]["inpatient_beds_used_covid"]
                * self.hospitalizations_by_age_vaccination["BiV_Multiplier"]
            ).astype(int)
            self.epidemiological_data[state]["Hospitalized_BiV"] = cdc_skew

            for vaccination_group in vaccination_groups:
                for age_group in age_groups:
                    self.epidemiological_data[state][
                        f"Hospitalized_{age_group}_{vaccination_group}"
                    ] = (
                        self.epidemiological_data[state]["inpatient_beds_used_covid"]
                        * self.hospitalizations_by_age_vaccination[
                            f"{age_group}_{vaccination_group}_Multiplier"
                        ]
                    ).astype(
                        int
                    )

            # Recovered compartments.
            # Computing the recoveries by vaccination groups.
            initial_recovered_unvaccinated_skew = (
                self.epidemiological_data[state]["Total Recovered"].iloc[0]
                * self.cases_by_age_vaccination["UV_Multiplier"].iloc[0]
            ).astype(int)

            initial_recovered_fully_vaccinated_skew = (
                self.epidemiological_data[state]["Total Recovered"].iloc[0]
                * self.cases_by_age_vaccination["V_Multiplier"].iloc[0]
            ).astype(int)

            initial_recovered_booster_vaccinated_skew = (
                self.epidemiological_data[state]["Total Recovered"].iloc[0]
                * self.cases_by_age_vaccination["BiV_Multiplier"].iloc[0]
            ).astype(int)

            # Computing the recoveries by age groups.
            initial_recovered_5_17_skew = (
                self.epidemiological_data[state]["Total Recovered"].iloc[0]
                * self.cases_by_age_vaccination["5-17_Multiplier"].iloc[0]
            ).astype(int)

            initial_recovered_18_49_skew = (
                self.epidemiological_data[state]["Total Recovered"].iloc[0]
                * self.cases_by_age_vaccination["18-49_Multiplier"].iloc[0]
            ).astype(int)
            initial_recovered_50_64_skew = (
                self.epidemiological_data[state]["Total Recovered"].iloc[0]
                * self.cases_by_age_vaccination["50-64_Multiplier"].iloc[0]
            ).astype(int)
            initial_recovered_65_plus_skew = (
                self.epidemiological_data[state]["Total Recovered"].iloc[0]
                * self.cases_by_age_vaccination["65+_Multiplier"].iloc[0]
            ).astype(int)

            uv_to_v = self.epidemiological_data[state][
                "percentage_unvaccinated_to_vaccinated"
            ]
            v_to_biv = self.epidemiological_data[state][
                "percentage_vaccinated_to_bivalent_vaccinated"
            ]

            self.epidemiological_data[state][
                [
                    "Recovered_UV",
                    "Recovered_V",
                    "Recovered_BiV",
                    "Recovered_5-17",
                    "Recovered_18-49",
                    "Recovered_50-64",
                    "Recovered_65+",
                ]
            ] = 0

            self.epidemiological_data[state]["Recovered_UV"].iloc[
                0
            ] = initial_recovered_unvaccinated_skew
            self.epidemiological_data[state]["Recovered_V"].iloc[
                0
            ] = initial_recovered_fully_vaccinated_skew
            self.epidemiological_data[state]["Recovered_BiV"].iloc[
                0
            ] = initial_recovered_booster_vaccinated_skew

            self.epidemiological_data[state]["Recovered_5-17"].iloc[
                0
            ] = initial_recovered_5_17_skew
            self.epidemiological_data[state]["Recovered_18-49"].iloc[
                0
            ] = initial_recovered_18_49_skew
            self.epidemiological_data[state]["Recovered_50-64"].iloc[
                0
            ] = initial_recovered_50_64_skew
            self.epidemiological_data[state]["Recovered_65+"].iloc[
                0
            ] = initial_recovered_65_plus_skew

            for i in range(1, len(self.epidemiological_data[state])):
                # Computing the Recovered compartments by vaccination groups.
                self.epidemiological_data[state]["Recovered_UV"].iloc[i] = (
                    self.epidemiological_data[state]["Recovered_UV"].iloc[i - 1]
                    + self.epidemiological_data[state]["New Recoveries"].iloc[i]
                    * self.cases_by_age_vaccination["UV_Multiplier"].iloc[i]
                    - uv_to_v[i]
                    * self.epidemiological_data[state]["Recovered_UV"].iloc[i - 1]
                ).astype(int)
                self.epidemiological_data[state]["Recovered_V"].iloc[i] = (
                    self.epidemiological_data[state]["Recovered_V"].iloc[i - 1]
                    + self.epidemiological_data[state]["New Recoveries"].iloc[i]
                    * self.cases_by_age_vaccination["V_Multiplier"].iloc[i]
                    + uv_to_v[i]
                    * self.epidemiological_data[state]["Recovered_UV"].iloc[i - 1]
                    - v_to_biv[i]
                    * self.epidemiological_data[state]["Recovered_V"].iloc[i - 1]
                ).astype(int)
                self.epidemiological_data[state]["Recovered_BiV"].iloc[i] = (
                    self.epidemiological_data[state]["Recovered_BiV"].iloc[i - 1]
                    + self.epidemiological_data[state]["New Recoveries"].iloc[i]
                    * self.cases_by_age_vaccination["BiV_Multiplier"].iloc[i]
                    + v_to_biv[i]
                    * self.epidemiological_data[state]["Recovered_V"].iloc[i - 1]
                ).astype(int)

                # Computing the Recovered compartments by age groups.
                self.epidemiological_data[state]["Recovered_5-17"].iloc[i] = (
                    self.epidemiological_data[state]["Recovered_5-17"].iloc[i - 1]
                    + self.epidemiological_data[state]["New Recoveries"].iloc[i]
                    * self.cases_by_age_vaccination["5-17_Multiplier"].iloc[i]
                ).astype(int)
                self.epidemiological_data[state]["Recovered_18-49"].iloc[i] = (
                    self.epidemiological_data[state]["Recovered_18-49"].iloc[i - 1]
                    + self.epidemiological_data[state]["New Recoveries"].iloc[i]
                    * self.cases_by_age_vaccination["18-49_Multiplier"].iloc[i]
                ).astype(int)
                self.epidemiological_data[state]["Recovered_50-64"].iloc[i] = (
                    self.epidemiological_data[state]["Recovered_50-64"].iloc[i - 1]
                    + self.epidemiological_data[state]["New Recoveries"].iloc[i]
                    * self.cases_by_age_vaccination["50-64_Multiplier"].iloc[i]
                ).astype(int)
                self.epidemiological_data[state]["Recovered_65+"].iloc[i] = (
                    self.epidemiological_data[state]["Recovered_65+"].iloc[i - 1]
                    + self.epidemiological_data[state]["New Recoveries"].iloc[i]
                    * self.cases_by_age_vaccination["65+_Multiplier"].iloc[i]
                ).astype(int)

            for vaccination_group in vaccination_groups:
                for age_group in age_groups:
                    self.epidemiological_data[state][
                        f"Recovered_{age_group}_{vaccination_group}"
                    ] = (
                        # self.epidemiological_data[state][
                        #     f"Recovered_{vaccination_group}"
                        # ]
                        self.epidemiological_data[state][f"Recovered_{age_group}"]
                        * self.cases_by_age_vaccination[
                            f"{age_group}_{vaccination_group}_Multiplier"
                        ]
                    ).astype(
                        int
                    )

            # Deceased compartments.
            # Computing the Deceased compartment values by vaccination groups.
            initial_deceased_skew = (
                self.epidemiological_data[state]["Total Deaths (Linear)"].iloc[0]
                * self.deaths_by_age_vaccination["UV_Multiplier"].iloc[0]
            ).astype(int)
            cdc_skew = (
                self.epidemiological_data[state]["Daily Deaths"]
                * self.deaths_by_age_vaccination["UV_Multiplier"]
            ).astype(int)
            cdc_skew[0] = 0
            cdc_skew = cdc_skew.cumsum()
            cdc_skew = cdc_skew + initial_deceased_skew
            self.epidemiological_data[state]["Deceased_UV"] = cdc_skew

            initial_deceased_skew = (
                self.epidemiological_data[state]["Total Deaths (Linear)"].iloc[0]
                * self.deaths_by_age_vaccination["V_Multiplier"].iloc[0]
            ).astype(int)
            cdc_skew = (
                self.epidemiological_data[state]["Daily Deaths"]
                * self.deaths_by_age_vaccination["V_Multiplier"]
            ).astype(int)
            cdc_skew[0] = 0
            cdc_skew = cdc_skew.cumsum()
            cdc_skew = cdc_skew + initial_deceased_skew
            self.epidemiological_data[state]["Deceased_V"] = cdc_skew

            initial_deceased_skew = (
                self.epidemiological_data[state]["Total Deaths (Linear)"].iloc[0]
                * self.deaths_by_age_vaccination["BiV_Multiplier"].iloc[0]
            ).astype(int)
            cdc_skew = (
                self.epidemiological_data[state]["Daily Deaths"]
                * self.deaths_by_age_vaccination["BiV_Multiplier"]
            ).astype(int)
            cdc_skew[0] = 0
            cdc_skew = cdc_skew.cumsum()
            cdc_skew = cdc_skew + initial_deceased_skew
            self.epidemiological_data[state]["Deceased_BiV"] = cdc_skew

            # Computing the Deceased compartment values by age groups.
            initial_deceased_skew = (
                self.epidemiological_data[state]["Total Deaths (Linear)"].iloc[0]
                * self.deaths_by_age_vaccination["5-17_Multiplier"].iloc[0]
            ).astype(int)
            cdc_skew = (
                self.epidemiological_data[state]["Daily Deaths"]
                * self.deaths_by_age_vaccination["5-17_Multiplier"]
            ).astype(int)
            cdc_skew[0] = 0
            cdc_skew = cdc_skew.cumsum()
            cdc_skew = cdc_skew + initial_deceased_skew
            self.epidemiological_data[state]["Deceased_5-17"] = cdc_skew
            self.epidemiological_data[state]["Deceased_5-17"].replace(
                0, method="ffill", inplace=True
            )

            initial_deceased_skew = (
                self.epidemiological_data[state]["Total Deaths (Linear)"].iloc[0]
                * self.deaths_by_age_vaccination["18-49_Multiplier"].iloc[0]
            ).astype(int)
            cdc_skew = (
                self.epidemiological_data[state]["Daily Deaths"]
                * self.deaths_by_age_vaccination["18-49_Multiplier"]
            ).astype(int)
            cdc_skew[0] = 0
            cdc_skew = cdc_skew.cumsum()
            cdc_skew = cdc_skew + initial_deceased_skew
            self.epidemiological_data[state]["Deceased_18-49"] = cdc_skew
            self.epidemiological_data[state]["Deceased_18-49"].replace(
                0, method="ffill", inplace=True
            )

            initial_deceased_skew = (
                self.epidemiological_data[state]["Total Deaths (Linear)"].iloc[0]
                * self.deaths_by_age_vaccination["50-64_Multiplier"].iloc[0]
            ).astype(int)
            cdc_skew = (
                self.epidemiological_data[state]["Daily Deaths"]
                * self.deaths_by_age_vaccination["50-64_Multiplier"]
            ).astype(int)
            cdc_skew[0] = 0
            cdc_skew = cdc_skew.cumsum()
            cdc_skew = cdc_skew + initial_deceased_skew
            self.epidemiological_data[state]["Deceased_50-64"] = cdc_skew
            self.epidemiological_data[state]["Deceased_50-64"].replace(
                0, method="ffill", inplace=True
            )

            initial_deceased_skew = (
                self.epidemiological_data[state]["Total Deaths (Linear)"].iloc[0]
                * self.deaths_by_age_vaccination["65+_Multiplier"].iloc[0]
            ).astype(int)
            cdc_skew = (
                self.epidemiological_data[state]["Daily Deaths"]
                * self.deaths_by_age_vaccination["65+_Multiplier"]
            ).astype(int)
            cdc_skew[0] = 0
            cdc_skew = cdc_skew.cumsum()
            cdc_skew = cdc_skew + initial_deceased_skew
            self.epidemiological_data[state]["Deceased_65+"] = cdc_skew
            self.epidemiological_data[state]["Deceased_65+"].replace(
                0, method="ffill", inplace=True
            )

            for vaccination_group in vaccination_groups:
                for age_group in age_groups:
                    self.epidemiological_data[state][
                        f"Deceased_{age_group}_{vaccination_group}"
                    ] = (
                        # self.epidemiological_data[state][
                        #     f"Deceased_{vaccination_group}"
                        # ]
                        self.epidemiological_data[state][f"Deceased_{age_group}"]
                        * self.deaths_by_age_vaccination[
                            f"{age_group}_{vaccination_group}_Multiplier"
                        ]
                    ).astype(
                        int
                    )
                    self.epidemiological_data[state][
                        f"Deceased_{age_group}_{vaccination_group}"
                    ].replace(0, method="ffill", inplace=True)

            # Accounting for "missing individuals".
            missing_individuals_unvaccinated = self.epidemiological_data[state][
                "unvaccinated_individuals"
            ] - (
                self.epidemiological_data[state]["Infected_UV"]
                + self.epidemiological_data[state]["Hospitalized_UV"]
                + self.epidemiological_data[state]["Recovered_UV"]
                + self.epidemiological_data[state]["Deceased_UV"]
            )

            missing_individuals_vaccinated = self.epidemiological_data[state][
                "vaccinated_individuals"
            ] - (
                self.epidemiological_data[state]["Infected_V"]
                + self.epidemiological_data[state]["Hospitalized_V"]
                + self.epidemiological_data[state]["Recovered_V"]
                + self.epidemiological_data[state]["Deceased_V"]
            )

            missing_individuals_bivalent_booster_vaccinated = self.epidemiological_data[
                state
            ]["bivalent_booster_vaccinated_individuals"] - (
                self.epidemiological_data[state]["Infected_BiV"]
                + self.epidemiological_data[state]["Hospitalized_BiV"]
                + self.epidemiological_data[state]["Recovered_BiV"]
                + self.epidemiological_data[state]["Deceased_BiV"]
            )

            total_missing_individuals_vaccination = (
                missing_individuals_unvaccinated
                + missing_individuals_vaccinated
                + missing_individuals_bivalent_booster_vaccinated
            )

            self.epidemiological_data[state]["Infected"] = (
                self.epidemiological_data[state]["Infected_UV"]
                + self.epidemiological_data[state]["Infected_V"]
                + self.epidemiological_data[state]["Infected_BiV"]
            )

            self.epidemiological_data[state]["Hospitalized"] = (
                self.epidemiological_data[state]["Hospitalized_UV"]
                + self.epidemiological_data[state]["Hospitalized_V"]
                + self.epidemiological_data[state]["Hospitalized_BiV"]
            )

            self.epidemiological_data[state]["Recovered"] = (
                self.epidemiological_data[state]["Recovered_UV"]
                + self.epidemiological_data[state]["Recovered_V"]
                + self.epidemiological_data[state]["Recovered_BiV"]
            )

            self.epidemiological_data[state]["Deceased"] = (
                self.epidemiological_data[state]["Deceased_UV"]
                + self.epidemiological_data[state]["Deceased_V"]
                + self.epidemiological_data[state]["Deceased_BiV"]
            )

            # Adjusting Susceptible
            self.epidemiological_data[state]["Susceptible_UV"] = (
                self.epidemiological_data[state]["Susceptible"]
                * missing_individuals_unvaccinated
                / total_missing_individuals_vaccination
            ).astype(int)

            self.epidemiological_data[state]["Susceptible_V"] = (
                self.epidemiological_data[state]["Susceptible"]
                * missing_individuals_vaccinated
                / total_missing_individuals_vaccination
            ).astype(int)

            self.epidemiological_data[state]["Susceptible_BiV"] = (
                self.epidemiological_data[state]["Susceptible"]
                * missing_individuals_bivalent_booster_vaccinated
                / total_missing_individuals_vaccination
            ).astype(int)

            for vaccination_group in vaccination_groups:
                for age_group in age_groups:
                    self.epidemiological_data[state][
                        f"Susceptible_{age_group}_{vaccination_group}"
                    ] = (
                        self.epidemiological_data[state][
                            f"Susceptible_{vaccination_group}"
                        ]
                        * (
                            self.cases_by_age_vaccination[f"{age_group}_Population"]
                            / (
                                self.cases_by_age_vaccination["5-17_Population"]
                                + self.cases_by_age_vaccination["18-49_Population"]
                                + self.cases_by_age_vaccination["50-64_Population"]
                                + self.cases_by_age_vaccination["65+_Population"]
                            )
                        )
                    ).astype(
                        int
                    )

            # # Adjusting Exposed
            # self.epidemiological_data[state]["Exposed_UV"] = (
            #     self.epidemiological_data[state]["Exposed"]
            #     * missing_individuals_unvaccinated
            #     / total_missing_individuals_vaccination
            # ).astype(int)
            #
            # self.epidemiological_data[state]["Exposed_V"] = (
            #     self.epidemiological_data[state]["Exposed"]
            #     * missing_individuals_vaccinated
            #     / total_missing_individuals_vaccination
            # ).astype(int)
            #
            # self.epidemiological_data[state]["Exposed_BiV"] = (
            #     self.epidemiological_data[state]["Exposed"]
            #     * missing_individuals_bivalent_booster_vaccinated
            #     / total_missing_individuals_vaccination
            # ).astype(int)

            # for vaccination_group in vaccination_groups:
            #     for age_group in age_groups:
            #         self.epidemiological_data[state][
            #             f"Exposed_{age_group}_{vaccination_group}"
            #         ] = (
            #             self.epidemiological_data[state][f"Exposed_{vaccination_group}"]
            #             * (
            #                 self.cases_by_age_vaccination[f"{age_group}_Population"]
            #                 / (
            #                     self.cases_by_age_vaccination["5-17_Population"]
            #                     + self.cases_by_age_vaccination["18-49_Population"]
            #                     + self.cases_by_age_vaccination["50-64_Population"]
            #                     + self.cases_by_age_vaccination["65+_Population"]
            #                 )
            #             )
            #         ).astype(
            #             int
            #         )

            # Detected
            cdc_skew = (
                self.epidemiological_data[state]["new_positive_tests"]
                * (self.cases_by_age_vaccination["UV_Multiplier"])
            ).astype(int)
            self.epidemiological_data[state]["Detected_UV"] = cdc_skew

            cdc_skew = (
                self.epidemiological_data[state]["new_positive_tests"]
                * (self.cases_by_age_vaccination["V_Multiplier"])
            ).astype(int)
            self.epidemiological_data[state]["Detected_V"] = cdc_skew

            cdc_skew = (
                self.epidemiological_data[state]["new_positive_tests"]
                * (self.cases_by_age_vaccination["BiV_Multiplier"])
            ).astype(int)
            self.epidemiological_data[state]["Detected_BiV"] = cdc_skew

            for vaccination_group in vaccination_groups:
                for age_group in age_groups:
                    self.epidemiological_data[state][
                        f"Detected_{age_group}_{vaccination_group}"
                    ] = (
                        self.epidemiological_data[state]["new_positive_tests"]
                        * (self.cases_by_age_vaccination[f"{age_group}_Multiplier"])
                        * self.cases_by_age_vaccination[
                            f"{age_group}_{vaccination_group}_Multiplier"
                        ]
                    ).astype(
                        int
                    )

            self.epidemiological_data[state]["Detected"] = (
                self.epidemiological_data[state]["Detected_UV"]
                + self.epidemiological_data[state]["Detected_V"]
                + self.epidemiological_data[state]["Detected_BiV"]
            )

            # Computing the total by vaccination statues across the different compartments.
            self.epidemiological_data[state]["unvaccinated_compartment_total"] = (
                self.epidemiological_data[state]["Susceptible_UV"]
                # + self.epidemiological_data[state]["Exposed_UV"]
                + self.epidemiological_data[state]["Infected_UV"]
                + self.epidemiological_data[state]["Hospitalized_UV"]
                + self.epidemiological_data[state]["Recovered_UV"]
                + self.epidemiological_data[state]["Deceased_UV"]
            )

            self.epidemiological_data[state]["vaccinated_compartment_total"] = (
                self.epidemiological_data[state]["Susceptible_V"]
                # + self.epidemiological_data[state]["Exposed_V"]
                + self.epidemiological_data[state]["Infected_V"]
                + self.epidemiological_data[state]["Hospitalized_V"]
                + self.epidemiological_data[state]["Recovered_V"]
                + self.epidemiological_data[state]["Deceased_V"]
            )

            self.epidemiological_data[state][
                "bivalent_booster_vaccinated_compartment_total"
            ] = (
                self.epidemiological_data[state]["Susceptible_BiV"]
                # + self.epidemiological_data[state]["Exposed_BiV"]
                + self.epidemiological_data[state]["Infected_BiV"]
                + self.epidemiological_data[state]["Hospitalized_BiV"]
                + self.epidemiological_data[state]["Recovered_BiV"]
                + self.epidemiological_data[state]["Deceased_BiV"]
            )

            self.epidemiological_data[state]["Original Infected"] = (
                self.epidemiological_data[state]["Active Cases"]
                - self.epidemiological_data[state]["inpatient_beds_used_covid"]
            )

            columns_to_add = []
            compartments = [
                "Susceptible",
                # "Exposed",
                "Infected",
                "Hospitalized",
                "Recovered",
                "Deceased",
                "Detected",
            ]
            test_and_mobility_columns = [
                "new_tests",
                "total_tests",
                "new_negative_tests",
                "total_negative_tests",
                "new_positive_tests",
                "total_positive_tests",
                "new_inconclusive_tests",
                "total_inconclusive_tests",
                "retail_and_recreation_percent_change_from_baseline",
                "grocery_and_pharmacy_percent_change_from_baseline",
                "parks_percent_change_from_baseline",
                "transit_stations_percent_change_from_baseline",
                "workplaces_percent_change_from_baseline",
                "residential_percent_change_from_baseline",
            ]

            for i in range(560, len(self.epidemiological_data[state]) - 7, 7):
                self.epidemiological_data[state].loc[i:i+6, ["retail_and_recreation_percent_change_from_baseline",
                    "grocery_and_pharmacy_percent_change_from_baseline",
                    "parks_percent_change_from_baseline",
                    "transit_stations_percent_change_from_baseline",
                    "workplaces_percent_change_from_baseline",
                    "residential_percent_change_from_baseline"]] = self.epidemiological_data[state][["retail_and_recreation_percent_change_from_baseline",
                    "grocery_and_pharmacy_percent_change_from_baseline",
                    "parks_percent_change_from_baseline",
                    "transit_stations_percent_change_from_baseline",
                    "workplaces_percent_change_from_baseline",
                    "residential_percent_change_from_baseline"]].iloc[553:560].values
                # print(i)
            self.epidemiological_data[state].loc[721:726, ["retail_and_recreation_percent_change_from_baseline",
                                                           "grocery_and_pharmacy_percent_change_from_baseline",
                                                           "parks_percent_change_from_baseline",
                                                           "transit_stations_percent_change_from_baseline",
                                                           "workplaces_percent_change_from_baseline",
                                                           "residential_percent_change_from_baseline"]] = \
            self.epidemiological_data[state][["retail_and_recreation_percent_change_from_baseline",
                                              "grocery_and_pharmacy_percent_change_from_baseline",
                                              "parks_percent_change_from_baseline",
                                              "transit_stations_percent_change_from_baseline",
                                              "workplaces_percent_change_from_baseline",
                                              "residential_percent_change_from_baseline"]].iloc[553:559].values

            for compartment in compartments:
                for age_group in age_groups:
                    for vaccination_group in vaccination_groups:
                        columns_to_add.append(
                            f"{compartment}_{age_group}_{vaccination_group}"
                        )

            # Saving the epidemiological model data.
            self.epidemiological_data[state].iloc[:].to_csv(
                f"{data_directory}/epidemiological_model_data/{state}.csv",
                index=False,
                columns=[
                    "date",
                    "unvaccinated_individuals",
                    "vaccinated_individuals",
                    "bivalent_booster_vaccinated_individuals",
                    "unvaccinated_compartment_total",
                    "vaccinated_compartment_total",
                    "bivalent_booster_vaccinated_compartment_total",
                    "percentage_unvaccinated_to_vaccinated",
                    "percentage_vaccinated_to_bivalent_vaccinated",
                    "Daily Cases",
                    "Susceptible",
                    # "Exposed",
                    "Infected",
                    "Hospitalized",
                    "Recovered",
                    "Deceased",
                    "Detected",
                    # "Original Infected",
                    # "inpatient_beds_used_covid",
                    # "Total Recovered",
                    # "Total Deaths (Linear)",
                    "Susceptible_UV",
                    "Susceptible_V",
                    "Susceptible_BiV",
                    # "Exposed_UV",
                    # "Exposed_V",
                    # "Exposed_BiV",
                    "Infected_UV",
                    "Infected_V",
                    "Infected_BiV",
                    "Hospitalized_UV",
                    "Hospitalized_V",
                    "Hospitalized_BiV",
                    "Recovered_UV",
                    "Recovered_V",
                    "Recovered_BiV",
                    "Deceased_UV",
                    "Deceased_V",
                    "Deceased_BiV",
                    "Detected_UV",
                    "Detected_V",
                    "Detected_BiV",
                ]
                + columns_to_add
                + test_and_mobility_columns,
            )


data__paths = {
    "processed_state_data": f"{data_directory}/processed_state_data/",
    "cases_by_age_vaccination": f"{data_directory}/data_by_age_vaccination_status/cases_by_age_vaccination_all.csv",
    "deaths_by_age_vaccination": f"{data_directory}/data_by_age_vaccination_status/deaths_by_age_vaccination_all.csv",
    "hospitalizations_by_age_vaccination": f"{data_directory}/data_by_age_vaccination_status/"
    f"hospitalization_by_age_vaccination.csv",
}

epidemiological_data_preprocessing = EpidemiologicalDataPreProcessing(
    data_paths=data__paths
)
epidemiological_data_preprocessing.compute_age_group_and_vaccination_status_multipliers()
epidemiological_data_preprocessing.data_preprocessing()
