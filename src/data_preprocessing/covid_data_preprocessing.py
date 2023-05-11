import json

import epiweeks
import numpy as np
import pandas as pd

from src.settings import data_directory
from src.utilities.parameter_initializer import ParameterInitializer

pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None


# noinspection DuplicatedCode
class DataPreprocessing:
    """This class preprocesses the data for the country and creates the state data."""

    def __init__(self, data_paths):
        """This method reads all the data files and creates the state mapping."""

        self.data_paths = data_paths
        self.parameter_initializer = ParameterInitializer(
            data_paths["cases_and_outcomes"]
        )

        self.states = self.parameter_initializer.initialize_state_names()

        self.state_name_to_state_abbreviation = json.load(
            open(data_paths["state_abbreviations"])
        )

        self.us_testing = pd.read_csv(data_paths["testing"])
        self.us_hospitalizations = pd.read_csv(data_paths["hospitalization"])
        self.us_vaccinations = pd.read_csv(data_paths["vaccination"])
        self.google_mobility_report = pd.read_csv(data_paths["google_mobility_report"])
        self.cases_deaths_by_age_vaccination = pd.read_csv(
            data_paths["cases_deaths_by_age_vaccination"]
        )
        self.cases_deaths_by_age_first_booster = pd.read_csv(
            data_paths["cases_deaths_by_age_booster"]
        )
        self.cases_deaths_by_age_second_booster = pd.read_csv(
            data_paths["cases_deaths_by_age_second_booster"]
        )
        self.cases_deaths_by_age_bivalent_booster = pd.read_csv(
            data_paths["cases_deaths_by_age_bivalent_booster"]
        )
        self.hospitalizations_by_vaccination = pd.read_csv(
            self.data_paths["hospitalizations_by_vaccination"]
        )

    @staticmethod
    def data_imputer(data, imputation_columns, imputation_method="same"):
        """This method imputes the missing data.
        :param data: Pandas Dataframe - Dataframe for which data is to be imputed.
        :param imputation_columns: List[String] - Column names in data for which data is to be imputed.
        :param imputation_method: String - Imputation technique to impute the missing data. Can be
            1. "same" - Imputes the same values as the next available value.
            2. "difference". Calculates the difference between the last value and the next available value. We then
                             evenly split this difference between the missing entries.
        """

        if imputation_method == "same":
            for column_name in imputation_columns:
                for i in range(len(data) - 0):
                    if pd.isnull(data[column_name][i]):
                        counter = 1

                        try:
                            while pd.isnull(data[column_name][i + counter]):
                                counter += 1
                        except KeyError:
                            pass

                        if counter >= 7:
                            for j in range(counter):
                                data[column_name][i + j] = np.NAN
                        else:
                            for j in range(counter):
                                try:
                                    data[column_name][i + j] = data[column_name][i + counter]
                                except KeyError:
                                    pass

        elif imputation_method == "difference":
            print("TODO")

        return data

    def create_state_vaccination_data(self):
        """This method takes the U.S. vaccination data and creates the state vaccination data."""

        for state in self.states:
            state_vaccination = self.us_vaccinations.loc[
                self.us_vaccinations["Location"]
                == self.state_name_to_state_abbreviation[state]
                ]

            state_vaccination.rename(
                columns={"Date": "date", "Location": "state"}, inplace=True
            )
            state_vaccination["date"] = pd.to_datetime(state_vaccination["date"])
            state_vaccination = state_vaccination.sort_values(by="date")
            state_vaccination = state_vaccination.reset_index(drop=True, inplace=False)
            state_vaccination = state_vaccination.fillna(0)

            # Beginning June 13, 2022, instead of daily, jurisdictions and other partners report vaccine
            # administration and delivery data to CDC weekly on Wednesdays by 6 AM ET. Thus, we will impute the daily
            # data with an even split of the difference between successive weeks.
            earliest_date = state_vaccination["date"].iloc[0]
            latest_date = state_vaccination["date"].iloc[-1]

            dates = pd.date_range(earliest_date, latest_date, freq="d")
            dates = pd.DataFrame(dates, columns=["date"])
            state_vaccination = dates.merge(state_vaccination, how="outer", on="date")

            # We handle the intermediary missing data values by calculating the difference between the last and
            # the next date for which the data is available. We then evenly split this difference between the missing
            # entries.
            for i in range(1, len(state_vaccination)):
                for column_name in state_vaccination.columns:
                    if type(state_vaccination[column_name][0]) == str:
                        continue

                    elif pd.isnull(state_vaccination[column_name][i]):
                        counter = 1

                        while pd.isnull(state_vaccination[column_name][i + counter]):
                            counter += 1

                        difference = (
                                state_vaccination[column_name][i + counter]
                                - state_vaccination[column_name][i - 1]
                        )

                        for j in range(counter):
                            if state_vaccination[column_name][i - 1].is_integer():
                                state_vaccination[column_name][i + j] = int(
                                    state_vaccination[column_name][i + j - 1]
                                    + difference / (counter + 1)
                                )
                            elif not state_vaccination[column_name][i - 1].is_integer():
                                state_vaccination[column_name][i + j] = float(
                                    state_vaccination[column_name][i + j - 1]
                                    + difference / (counter + 1)
                                )

            state_vaccination["state"] = [state for _ in range(len(state_vaccination))]

            state_vaccination.to_csv(
                f"{data_directory}/vaccinations/{state}.csv",
                index=False,
            )

    def create_state_hospitalization_data(self):
        """This method takes the U.S. hospitalization data and creates the state hospitalization data."""

        for state in self.states:
            state_hospitalization = self.us_hospitalizations.loc[
                self.us_hospitalizations["state"]
                == self.state_name_to_state_abbreviation[state]
                ]
            state_hospitalization = state_hospitalization.sort_values(by="date")
            state_hospitalization["state"] = [
                state for _ in range(len(state_hospitalization))
            ]

            state_hospitalization.to_csv(
                f"{data_directory}/hospitalization/{state}.csv",
                index=False,
            )

    def create_state_testing_data(self):
        """This method takes the U.S. testing data and creates the state testing data."""

        for state in self.states:
            state_testing = self.us_testing.loc[self.us_testing["state_name"] == state]

            state_dataframe = pd.DataFrame(
                columns=[
                    "date",
                    "state",
                    "new_tests",
                    "total_tests",
                    "new_negative_tests",
                    "total_negative_tests",
                    "new_positive_tests",
                    "total_positive_tests",
                    "new_inconclusive_tests",
                    "total_inconclusive_tests",
                ]
            )

            state_dataframe["date"] = state_testing["date"].unique()
            state_dataframe["state"] = [
                state_testing["state_name"].iloc[0]
                for _ in range(len(state_dataframe["date"]))
            ]

            (
                new_tests,
                total_tests,
                new_negative_tests,
                total_negative_tests,
                new_positive_tests,
                total_positive_tests,
                new_inconclusive_tests,
                total_inconclusive_tests,
            ) = ([], [], [], [], [], [], [], [])

            previous_date = None

            for date in state_testing["date"]:
                if date != previous_date:
                    new_tests.append(
                        sum(
                            state_testing.loc[state_testing["date"] == date][
                                "new_results_reported"
                            ]
                        )
                    )
                    total_tests.append(
                        sum(
                            state_testing.loc[state_testing["date"] == date][
                                "total_results_reported"
                            ]
                        )
                    )

                    temp = state_testing.loc[
                        (
                                (state_testing["date"] == date)
                                & (state_testing["overall_outcome"] == "Negative")
                        )
                    ]["new_results_reported"]
                    new_negative_tests.append(
                        0 if len(temp) == 0 else int(temp.iloc[0])
                    )

                    temp = state_testing.loc[
                        (
                                (state_testing["date"] == date)
                                & (state_testing["overall_outcome"] == "Positive")
                        )
                    ]["new_results_reported"]
                    new_positive_tests.append(
                        0 if len(temp) == 0 else int(temp.iloc[0])
                    )

                    temp = state_testing.loc[
                        (
                                (state_testing["date"] == date)
                                & (state_testing["overall_outcome"] == "Inconclusive")
                        )
                    ]["new_results_reported"]
                    new_inconclusive_tests.append(
                        0 if len(temp) == 0 else int(temp.iloc[0])
                    )

                    temp = state_testing.loc[
                        (
                                (state_testing["date"] == date)
                                & (state_testing["overall_outcome"] == "Negative")
                        )
                    ]["total_results_reported"]
                    total_negative_tests.append(
                        0 if len(temp) == 0 else int(temp.iloc[0])
                    )

                    temp = state_testing.loc[
                        (
                                (state_testing["date"] == date)
                                & (state_testing["overall_outcome"] == "Positive")
                        )
                    ]["total_results_reported"]
                    total_positive_tests.append(
                        0 if len(temp) == 0 else int(temp.iloc[0])
                    )

                    temp = state_testing.loc[
                        (
                                (state_testing["date"] == date)
                                & (state_testing["overall_outcome"] == "Inconclusive")
                        )
                    ]["total_results_reported"]
                    total_inconclusive_tests.append(
                        0 if len(temp) == 0 else int(temp.iloc[0])
                    )

                    previous_date = date

            state_dataframe["new_tests"] = new_tests
            state_dataframe["total_tests"] = total_tests
            state_dataframe["new_negative_tests"] = new_negative_tests
            state_dataframe["new_positive_tests"] = new_positive_tests
            state_dataframe["new_inconclusive_tests"] = new_inconclusive_tests
            state_dataframe["total_negative_tests"] = total_negative_tests
            state_dataframe["total_positive_tests"] = total_positive_tests
            state_dataframe["total_inconclusive_tests"] = total_inconclusive_tests

            state_dataframe.to_csv(
                f"{data_directory}/testing/{state}.csv",
                index=False,
            )

    def create_state_mobility_data(self):
        """This method takes Google's Global Mobility Report and creates the U.S. states mobility report."""

        us_mobility = self.google_mobility_report.loc[
            self.google_mobility_report["country_region"] == "United States"
            ]

        for state in self.states:
            state_dataframe = us_mobility.loc[us_mobility["sub_region_1"] == state]
            state_dataframe = state_dataframe.loc[
                pd.isnull(state_dataframe["sub_region_2"])
            ]
            state_dataframe.rename(columns={"sub_region_1": "state"}, inplace=True)
            columns = [
                "date",
                "state",
                "retail_and_recreation_percent_change_from_baseline",
                "grocery_and_pharmacy_percent_change_from_baseline",
                "parks_percent_change_from_baseline",
                "transit_stations_percent_change_from_baseline",
                "workplaces_percent_change_from_baseline",
                "residential_percent_change_from_baseline",
            ]
            state_dataframe[columns].to_csv(
                f"{data_directory}/mobility/{state}.csv",
                index=False,
            )

    def preprocess_cases_and_outcomes_data(self):
        """This method preprocess the cases and outcomes data."""

        for state in self.states:
            state_cases_and_outcomes = pd.read_csv(
                f"{self.data_paths['cases_and_outcomes']}{state}.csv"
            )
            total_cases = state_cases_and_outcomes["Total Cases (Linear)"]
            active_cases = state_cases_and_outcomes["Active Cases"]
            total_deaths = state_cases_and_outcomes["Total Deaths (Linear)"]

            # This list contains the total number of recovered people.
            total_recovered = [
                total_cases[i] - active_cases[i] - total_deaths[i]
                for i in range(len(total_cases))
            ]

            # This list contains the number of recovered people per day.
            new_recoveries = [
                total_recovered[i] - total_recovered[i - 1]
                for i in range(1, len(total_recovered))
            ]
            new_recoveries.insert(0, total_recovered[0])

            state_cases_and_outcomes["New Recoveries"] = new_recoveries
            state_cases_and_outcomes["Total Recovered"] = total_recovered

            state_cases_and_outcomes.to_csv(
                f"{data_directory}/cases_and_outcomes/{state}.csv",
                index=False,
            )

    def create_state_final_dataset(self):
        """This method creates the final state datasets."""

        for state in self.states:
            state_cases_and_outcomes = pd.read_csv(
                f"{self.data_paths['cases_and_outcomes']}/{state}.csv"
            )
            state_vaccinations = pd.read_csv(
                f"{data_directory}/vaccinations/{state}.csv"
            )
            state_hospitalizations = pd.read_csv(
                f"{data_directory}/hospitalization/{state}.csv"
            )
            state_testing = pd.read_csv(f"{data_directory}/testing/{state}.csv")

            state_mobility = pd.read_csv(f"{data_directory}/mobility/{state}.csv")

            state_cases_and_outcomes["date"] = pd.to_datetime(
                state_cases_and_outcomes["date"]
            )
            state_vaccinations["date"] = pd.to_datetime(state_vaccinations["date"])
            state_hospitalizations["date"] = pd.to_datetime(
                state_hospitalizations["date"]
            )
            state_testing["date"] = pd.to_datetime(state_testing["date"])
            state_mobility["date"] = pd.to_datetime(state_mobility["date"])

            # Dropping the "state" column to avoid duplicates.
            state_hospitalizations = self.drop_columns(
                state_hospitalizations, ["state"]
            )
            state_testing = self.drop_columns(state_testing, ["state"])
            state_mobility = self.drop_columns(state_mobility, ["state"])

            state_final = state_cases_and_outcomes.merge(
                state_vaccinations, how="inner", on="date"
            )

            state_final = state_final.merge(
                state_hospitalizations, how="inner", on="date"
            )
            state_final = state_final.merge(state_testing, how="inner", on="date")
            state_final = state_final.merge(state_mobility, how="inner", on="date")

            state_final.to_csv(
                f"{data_directory}/processed_state_data/{state}.csv", index=False
            )
            print("State:", state, "Shape:", state_final.shape)

    def preprocess_data_by_age_group_vaccination_status(self):
        """This method preprocesses the data on cases, deaths, and hospitalizations by age groups."""

        def helper_preprocess_data_by_age_group_vaccination_status(
                data,
                age_groups,
                vaccination_groups,
                earliest_date,
                latest_date,
                output_file_name,
        ):
            """
            This is a helper function to preprocess the data by age groups and vaccination status.
            :param data: Pandas Dataframe - Data on vaccinate Effectiveness and Breakthrough Surveillance.
            :param age_groups: List - List of age groups in the data.
            :param vaccination_groups: List - List of vaccination groups in the data.
            :param earliest_date: String - The earliest date for which the data is available.
            :param latest_date: String - The latest date for which the data is available.
            :param output_file_name: String - The name of the output file.
            """

            data.reset_index(inplace=True)

            imputation_columns = []

            for age_group in age_groups:
                for vaccination_group in vaccination_groups:
                    data[f"{age_group}_{vaccination_group}_multiplier"] = [
                        np.NAN for _ in range(len(data))
                    ]
                    imputation_columns.append(f"{age_group}_{vaccination_group}_multiplier")

            # Primary Series Vaccination
            if vaccination_groups == ["UV", "PSV"]:

                weeks = data["MMWR week"].unique()
                week_end_dates = pd.Series(weeks).apply(
                    lambda x: epiweeks.Week.fromstring(str(x)).enddate()
                )
                data["week_end_date"] = [np.NAN for _ in range(len(data))]
                for i in range(len(week_end_dates)):
                    data["week_end_date"].iloc[i] = week_end_dates.iloc[i]

                for i, week in enumerate(weeks):
                    week_age_groups = np.unique(data.loc[(data["MMWR week"] == week)]["Age group"])
                    for age_group in week_age_groups:
                        unvaccinated_primary_series_vaccinated_irr = (
                            data.loc[(data["MMWR week"] == week)
                                     & (data["Age group"] == age_group)]["Crude IRR"].iloc[0]
                            if age_group != "all_ages_adj"
                            else data.loc[(data["MMWR week"] == week)
                                          & (data["Age group"] == age_group)]["Age adjusted IRR"].iloc[0]
                        )
                        unvaccinated_multiplier = unvaccinated_primary_series_vaccinated_irr / (
                                unvaccinated_primary_series_vaccinated_irr + 1
                        )

                        data[f"{age_group}_UV_multiplier"].iloc[i] = unvaccinated_multiplier
                        primary_series_vaccinated_multiplier = (
                                1 - unvaccinated_multiplier
                        )
                        data[f"{age_group}_PSV_multiplier"].iloc[i] = primary_series_vaccinated_multiplier

            # First Booster Vaccination
            elif vaccination_groups == ["UV", "PSV", "BV1"]:
                weeks = data["mmwr_week"].unique()
                week_end_dates = pd.Series(weeks).apply(
                    lambda x: epiweeks.Week.fromstring(str(x)).enddate()
                )
                data["week_end_date"] = [np.NAN for _ in range(len(data))]
                for i in range(len(week_end_dates)):
                    data["week_end_date"].iloc[i] = week_end_dates.iloc[i]

                for i, week in enumerate(weeks):
                    week_age_groups = np.unique(data.loc[(data["mmwr_week"] == week)]["age_group"])
                    for age_group in week_age_groups:
                        unvaccinated_primary_series_vaccinated_irr = (
                            data.loc[(data["mmwr_week"] == week)
                                     & (data["age_group"] == age_group)]["crude_irr"].iloc[0]
                            if age_group != "all_ages"
                            else data.loc[(data["mmwr_week"] == week)
                                          & (data["age_group"] == age_group)]["age_adj_irr"].iloc[0]
                        )

                        unvaccinated_first_booster_vaccinated_irr = (
                            data.loc[(data["mmwr_week"] == week)
                                     & (data["age_group"] == age_group)]["crude_booster_irr"].iloc[0]
                            if age_group != "all_ages"
                            else data.loc[(data["mmwr_week"] == week)
                                          & (data["age_group"] == age_group)]["age_adj_booster_irr"].iloc[0]
                        )

                        unvaccinated_multiplier = 1 / (
                                1 + (1 / unvaccinated_primary_series_vaccinated_irr)
                                + (1 / unvaccinated_first_booster_vaccinated_irr)
                        )
                        data[f"{age_group}_UV_multiplier"].iloc[i] = unvaccinated_multiplier
                        primary_series_vaccinated_multiplier = 1 / (
                                unvaccinated_primary_series_vaccinated_irr
                                + 1
                                + (1 / (unvaccinated_first_booster_vaccinated_irr
                                        / unvaccinated_primary_series_vaccinated_irr))
                        )
                        data[f"{age_group}_PSV_multiplier"].iloc[
                            i
                        ] = primary_series_vaccinated_multiplier

                        first_booster_vaccinated_multiplier = 1 / (
                                unvaccinated_first_booster_vaccinated_irr
                                + (unvaccinated_first_booster_vaccinated_irr
                                   / unvaccinated_primary_series_vaccinated_irr)
                                + 1
                        )
                        data[f"{age_group}_BV1_multiplier"].iloc[i] = first_booster_vaccinated_multiplier

            # Second Booster Vaccination
            elif vaccination_groups == ["UV", "PSV", "BV1", "BV2"]:
                weeks = data["mmwr_week"].unique()
                week_end_dates = pd.Series(weeks).apply(
                    lambda x: epiweeks.Week.fromstring(str(x)).enddate()
                )
                data["week_end_date"] = [np.NAN for _ in range(len(data))]
                for i in range(len(week_end_dates)):
                    data["week_end_date"].iloc[i] = week_end_dates.iloc[i]

                for i, week in enumerate(weeks):
                    week_age_groups = np.unique(data.loc[(data["mmwr_week"] == week)]["age_group"])
                    for age_group in week_age_groups:
                        unvaccinated_primary_series_vaccinated_irr = (
                            data.loc[(data["mmwr_week"] == week)
                                     & (data["age_group"] == age_group)]["crude_irr"].iloc[0]
                            if age_group != "all_ages"
                            else data.loc[(data["mmwr_week"] == week)
                                          & (data["age_group"] == age_group)]["age_adj_vax_irr"].iloc[0]
                        )

                        unvaccinated_first_booster_vaccinated_irr = (
                            data.loc[(data["mmwr_week"] == week)
                                     & (data["age_group"] == age_group)]["crude_one_booster_irr"].iloc[0]
                            if age_group != "all_ages"
                            else data.loc[(data["mmwr_week"] == week)
                                          & (data["age_group"] == age_group)]["age_adj_one_booster_irr"].iloc[0]
                        )

                        unvaccinated_second_booster_vaccinated_irr = (
                            data.loc[(data["mmwr_week"] == week)
                                     & (data["age_group"] == age_group)]["crude_two_booster_irr"].iloc[0]
                            if age_group != "all_ages"
                            else data.loc[(data["mmwr_week"] == week)
                                          & (data["age_group"] == age_group)]["age_adj_two_booster_irr"].iloc[0]
                        )

                        unvaccinated_multiplier = 1 / (
                                1
                                + (1 / unvaccinated_primary_series_vaccinated_irr)
                                + (1 / unvaccinated_first_booster_vaccinated_irr)
                                + (1 / unvaccinated_second_booster_vaccinated_irr)
                        )
                        data[f"{age_group}_UV_multiplier"].iloc[i] = unvaccinated_multiplier

                        primary_series_vaccinated_multiplier = 1 / (
                                unvaccinated_primary_series_vaccinated_irr
                                + 1
                                + (1 / (unvaccinated_first_booster_vaccinated_irr
                                        / unvaccinated_primary_series_vaccinated_irr))
                                + (1 / (unvaccinated_second_booster_vaccinated_irr
                                        / unvaccinated_primary_series_vaccinated_irr))
                        )
                        data[f"{age_group}_PSV_multiplier"].iloc[
                            i
                        ] = primary_series_vaccinated_multiplier

                        first_booster_vaccinated_multiplier = 1 / (
                                unvaccinated_first_booster_vaccinated_irr
                                + (unvaccinated_first_booster_vaccinated_irr
                                   / unvaccinated_primary_series_vaccinated_irr)
                                + 1
                                + (unvaccinated_first_booster_vaccinated_irr
                                   / unvaccinated_second_booster_vaccinated_irr)
                        )
                        data[f"{age_group}_BV1_multiplier"].iloc[i] = first_booster_vaccinated_multiplier

                        second_booster_vaccinated_multiplier = 1 / (
                                unvaccinated_second_booster_vaccinated_irr
                                + (unvaccinated_second_booster_vaccinated_irr
                                   / unvaccinated_primary_series_vaccinated_irr)
                                + (unvaccinated_second_booster_vaccinated_irr
                                   / unvaccinated_first_booster_vaccinated_irr)
                                + 1
                        )
                        data[f"{age_group}_BV2_multiplier"].iloc[
                            i
                        ] = second_booster_vaccinated_multiplier

            # Bivalent Booster Vaccination
            elif vaccination_groups == ["UV", "V", "BiV"]:
                weeks = data["mmwr_week"].unique()
                week_end_dates = pd.Series(weeks).apply(
                    lambda x: epiweeks.Week.fromstring(str(x)).enddate()
                )

                data["week_end_date"] = [np.NAN for _ in range(len(data))]
                for i in range(len(week_end_dates)):
                    data["week_end_date"].iloc[i] = week_end_dates.iloc[i]

                for i, week in enumerate(weeks):
                    week_age_groups = np.unique(data.loc[(data["mmwr_week"] == week)]["age_group"])
                    for age_group in week_age_groups:
                        unvaccinated_vaccinated_irr = (
                            data.loc[
                                (data["mmwr_week"] == week)
                                & (data["age_group"] == age_group)
                                & (data["vaccination_status"] == "vaccinated")
                                ]["crude_irr"].iloc[0]
                            if age_group != "all_ages"
                            else data.loc[
                                (data["mmwr_week"] == week)
                                & (data["age_group"] == age_group)
                                & (data["vaccination_status"] == "vaccinated")
                                ]["age_adj_irr"].iloc[0]
                        )

                        unvaccinated_bivalent_booster_vaccinated_irr = (
                            data.loc[
                                (data["mmwr_week"] == week)
                                & (data["age_group"] == age_group)
                                & (data["vaccination_status"] == "vax with updated booster")
                                ]["crude_irr"].iloc[0]
                            if age_group != "all_ages"
                            else data.loc[
                                (data["mmwr_week"] == week)
                                & (data["age_group"] == age_group)
                                & (data["vaccination_status"] == "vax with updated booster")
                                ]["age_adj_irr"].iloc[0]
                        )

                        unvaccinated_multiplier = 1 / (
                                1
                                + (1 / unvaccinated_vaccinated_irr)
                                + (1 / unvaccinated_bivalent_booster_vaccinated_irr)
                        )
                        data[f"{age_group}_UV_multiplier"].iloc[i] = unvaccinated_multiplier

                        primary_series_vaccinated_multiplier = 1 / (
                                unvaccinated_vaccinated_irr
                                + 1
                                + (unvaccinated_vaccinated_irr / unvaccinated_bivalent_booster_vaccinated_irr)
                        )
                        data[f"{age_group}_V_multiplier"].iloc[i] = primary_series_vaccinated_multiplier

                        bivalent_booster_vaccinated_multiplier = 1 / (
                                unvaccinated_bivalent_booster_vaccinated_irr
                                + (unvaccinated_bivalent_booster_vaccinated_irr / unvaccinated_vaccinated_irr)
                                + 1
                        )
                        data[f"{age_group}_BiV_multiplier"].iloc[i] = bivalent_booster_vaccinated_multiplier

            data["week_end_date"] = pd.to_datetime(data["week_end_date"])

            earliest_date = earliest_date
            latest_date = latest_date

            dates = pd.date_range(earliest_date, latest_date, freq="d")
            dates = pd.DataFrame(dates, columns=["week_end_date"])
            data = dates.merge(data, how="outer", on="week_end_date")

            data.rename(columns={"week_end_date": "date"}, inplace=True)
            data = data[["date"] + imputation_columns].dropna(how="all")

            data = self.data_imputer(
                data=data,
                imputation_columns=imputation_columns,
                imputation_method="same",
            )

            data.to_csv(
                f"{data_directory}/data_by_age_vaccination_status/{output_file_name}",
                index=False,
            )

        # Cases Deaths by primary series vaccination.
        cases_by_age_vaccination = self.cases_deaths_by_age_vaccination.loc[
            (self.cases_deaths_by_age_vaccination["outcome"] == "case")
            & (self.cases_deaths_by_age_vaccination["Vaccine product"] == "all_types")
            ]
        helper_preprocess_data_by_age_group_vaccination_status(
            data=cases_by_age_vaccination,
            age_groups=np.unique(cases_by_age_vaccination["Age group"]),
            vaccination_groups=["UV", "PSV"],
            earliest_date="04/04/2021",
            latest_date="09/24/2022",
            output_file_name="cases_by_age_primary_series_vaccination.csv",
        )

        deaths_by_age_vaccination = self.cases_deaths_by_age_vaccination.loc[
            (self.cases_deaths_by_age_vaccination["outcome"] == "death")
            & (self.cases_deaths_by_age_vaccination["Vaccine product"] == "all_types")
            ]
        helper_preprocess_data_by_age_group_vaccination_status(
            data=deaths_by_age_vaccination,
            age_groups=np.unique(deaths_by_age_vaccination["Age group"]),
            vaccination_groups=["UV", "PSV"],
            earliest_date="04/04/2021",
            latest_date="09/03/2022",
            output_file_name="deaths_by_age_primary_series_vaccination.csv",
        )

        # Cases Deaths by first booster vaccination.
        cases_by_age_first_booster = self.cases_deaths_by_age_first_booster.loc[
            (self.cases_deaths_by_age_first_booster["outcome"] == "case")
            & (self.cases_deaths_by_age_first_booster["vaccine_product"] == "all_types")
            ]
        helper_preprocess_data_by_age_group_vaccination_status(
            data=cases_by_age_first_booster,
            age_groups=np.unique(cases_by_age_first_booster["age_group"]),
            vaccination_groups=["UV", "PSV", "BV1"],
            earliest_date="09/19/2021",
            latest_date="09/24/2022",
            output_file_name="cases_by_age_first_booster_vaccination.csv",
        )

        deaths_by_age_first_booster = self.cases_deaths_by_age_first_booster.loc[
            (self.cases_deaths_by_age_first_booster["outcome"] == "death")
            & (self.cases_deaths_by_age_first_booster["vaccine_product"] == "all_types")
            ]
        helper_preprocess_data_by_age_group_vaccination_status(
            data=deaths_by_age_first_booster,
            age_groups=np.unique(deaths_by_age_first_booster["age_group"]),
            vaccination_groups=["UV", "PSV", "BV1"],
            earliest_date="09/19/2021",
            latest_date="09/03/2022",
            output_file_name="deaths_by_age_first_booster_vaccination.csv",
        )

        # Cases Deaths by second booster vaccination.
        cases_by_age_second_booster = self.cases_deaths_by_age_second_booster.loc[
            (self.cases_deaths_by_age_second_booster["outcome"] == "case")
            & (self.cases_deaths_by_age_second_booster["vaccine_product"] == "all_types")
            ]
        helper_preprocess_data_by_age_group_vaccination_status(
            data=cases_by_age_second_booster,
            age_groups=np.unique(cases_by_age_second_booster["age_group"]),
            vaccination_groups=["UV", "PSV", "BV1", "BV2"],
            earliest_date="03/27/2022",
            latest_date="09/24/2022",
            output_file_name="cases_by_age_second_booster_vaccination.csv",
        )

        deaths_by_age_second_booster = self.cases_deaths_by_age_second_booster.loc[
            (self.cases_deaths_by_age_second_booster["outcome"] == "death")
            & (self.cases_deaths_by_age_second_booster["vaccine_product"] == "all_types")
            ]
        helper_preprocess_data_by_age_group_vaccination_status(
            data=deaths_by_age_second_booster,
            age_groups=np.unique(deaths_by_age_second_booster["age_group"]),
            vaccination_groups=["UV", "PSV", "BV1", "BV2"],
            earliest_date="03/27/2022",
            latest_date="09/03/2022",
            output_file_name="deaths_by_age_second_booster_vaccination.csv",
        )

        # Cases Deaths by bivalent booster vaccination.
        cases_by_age_bivalent_booster = self.cases_deaths_by_age_bivalent_booster.loc[
            (self.cases_deaths_by_age_bivalent_booster["outcome"] == "case")
            & (self.cases_deaths_by_age_bivalent_booster["mmwr_week"] >= 202238)
            ]
        helper_preprocess_data_by_age_group_vaccination_status(
            data=cases_by_age_bivalent_booster,
            age_groups=np.unique(cases_by_age_bivalent_booster["age_group"]),
            vaccination_groups=["UV", "V", "BiV"],
            earliest_date="09/18/2022",
            latest_date="03/25/2023",
            output_file_name="cases_by_age_bivalent_booster_vaccination.csv",
        )

        deaths_by_age_bivalent_booster = self.cases_deaths_by_age_bivalent_booster.loc[
            (self.cases_deaths_by_age_bivalent_booster["outcome"] == "death")
            & (self.cases_deaths_by_age_bivalent_booster["mmwr_week"] >= 202238)
            ]
        helper_preprocess_data_by_age_group_vaccination_status(
            data=deaths_by_age_bivalent_booster,
            age_groups=np.unique(deaths_by_age_bivalent_booster["age_group"]),
            vaccination_groups=["UV", "V", "BiV"],
            earliest_date="09/25/2022",
            latest_date="03/04/2023",
            output_file_name="deaths_by_age_bivalent_booster_vaccination.csv",
        )

    @staticmethod
    def drop_columns(data, columns):
        """This method drops the columns which aren't required."""

        data = data.drop(columns=columns)

        return data


data__paths = {
    "state_abbreviations": f"{data_directory}/miscellaneous/state_abbreviations.json",
    "testing": f"{data_directory}/us_department_of_health_and_human_services/COVID-19_Diagnostic_Laboratory_Testing"
               f"__PCR_Testing__Time_Series.csv",
    "hospitalization": f"{data_directory}/us_department_of_health_and_human_services/COVID-19_Reported_Patient_Impact"
                       f"_and_Hospital_Capacity_by_State_"
                       f"Timeseries__RAW_.csv",
    "vaccination": f"{data_directory}/cdc/vaccination_distribution_and_coverage/COVID-19_Vaccinations_in_the_"
                   f"United_States_Jurisdiction.csv",
    "google_mobility_report": f"{data_directory}/mobility/Google/Global_Mobility_Report.csv",
    "cases_and_outcomes": f"{data_directory}/cases_and_outcomes/",
    "cases_deaths_by_age_vaccination": f"{data_directory}/cdc/vaccination_effectiveness_and_breakthrough_surveillance/"
                                       f"cases_deaths_by_age_vaccination.csv",
    "cases_deaths_by_age_booster": f"{data_directory}/cdc/vaccination_effectiveness_and_breakthrough_surveillance/"
                                   f"cases_deaths_by_age_booster.csv",
    "cases_deaths_by_age_second_booster": f"{data_directory}/cdc/"
                                          f"vaccination_effectiveness_and_breakthrough_surveillance/"
                                          f"cases_deaths_by_age_second_booster.csv",
    "cases_deaths_by_age_bivalent_booster": f"{data_directory}/cdc/"
                                            f"vaccination_effectiveness_and_breakthrough_surveillance/"
                                            f"cases_deaths_by_age_bivalent_booster.csv",
    "hospitalizations_by_vaccination": f"{data_directory}/cdc/"
                                       f"vaccination_effectiveness_and_breakthrough_surveillance/"
                                       f"hospitalizations_by_vaccination.csv",
}

data_preprocessing = DataPreprocessing(data_paths=data__paths)
# data_preprocessing.create_state_vaccination_data()
# data_preprocessing.create_state_testing_data()
# data_preprocessing.create_state_hospitalization_data()
# data_preprocessing.create_state_mobility_data()
# data_preprocessing.preprocess_cases_and_outcomes_data()
# data_preprocessing.create_state_final_dataset()
data_preprocessing.preprocess_data_by_age_group_vaccination_status()
