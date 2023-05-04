import pandas as pd
from src.settings import data_directory

pd.set_option("display.max_rows", None)


class EpidemiologicalDataPreProcessing:
    """This class creates the epidemiological data to be used for computing the parameters of the epidemiological
    model."""

    def __init__(self, filepath="./New_York.csv", population=19_453_734):
        """This method loads the data for pre-processing.

        :parameter filepath: String - Filepath of the epidemic dataset.
        :parameter population: Integer - Population of the epidemic region."""

        self.epidemiological_data = pd.read_csv(filepath)
        self.epidemiological_data["date"] = pd.to_datetime(
            self.epidemiological_data["date"]
        )

        self.epidemiological_data = self.epidemiological_data.iloc[79:474]
        self.epidemiological_data.reset_index(inplace=True)

        self.cases_by_vaccination = pd.read_csv(
            f"{data_directory}/Old Data/data_by_vaccination_status/booster/cases_by_vaccination_and_booster.csv"
        ).iloc[:395]
        self.deaths_by_vaccination = pd.read_csv(
            f"{data_directory}/Old Data/data_by_vaccination_status/booster/deaths_by_vaccination_and_booster.csv"
        ).iloc[:395]
        self.hospitalizations_by_vaccination = pd.read_csv(
            f"{data_directory}/Old Data/data_by_vaccination_status/booster/hospitalizations_by_vaccination_and_booster.csv"
        ).iloc[:395]
        self.population = population

    def data_preprocessing(self):
        """This method pre-processes the data for the sub-compartments in the epidemiological model."""

        # Vaccination compartments.
        self.epidemiological_data["unvaccinated_individuals"] = (
            self.population - self.epidemiological_data["people_vaccinated"]
        )

        self.epidemiological_data[
            "fully_vaccinated_individuals"
        ] = self.epidemiological_data["people_fully_vaccinated"]

        self.epidemiological_data["boosted_individuals"] = self.epidemiological_data[
            "total_boosters"
        ]

        # Computing the vaccination rates.
        self.epidemiological_data[
            [
                "percentage_unvaccinated_to_fully_vaccinated",
                "percentage_fully_vaccinated_to_boosted",
            ]
        ] = 0

        for i in range(1, len(self.epidemiological_data)):
            # Unvaccinated to Fully Vaccinated.
            self.epidemiological_data[
                "percentage_unvaccinated_to_fully_vaccinated"
            ].iloc[i] = (
                self.epidemiological_data["unvaccinated_individuals"].iloc[i - 1]
                - self.epidemiological_data["unvaccinated_individuals"].iloc[i]
            ) / self.epidemiological_data[
                "unvaccinated_individuals"
            ].iloc[
                i - 1
            ]

            # Fully Vaccinated to Boosted.
            self.epidemiological_data["percentage_fully_vaccinated_to_boosted"].iloc[
                i
            ] = (
                self.epidemiological_data["total_boosters"].iloc[i]
                - self.epidemiological_data["total_boosters"].iloc[i - 1]
            ) / self.epidemiological_data[
                "fully_vaccinated_individuals"
            ].iloc[
                i - 1
            ]

        # Exposed compartments.
        exposure_multiplier = (
            100 / 0.7
        )  # We have a reference for this. (Cited > 700 times).
        self.epidemiological_data["Exposed"] = (
            self.epidemiological_data["New Cases"] * exposure_multiplier
        ).astype(int)

        # Susceptible compartments.
        self.epidemiological_data["Susceptible"] = (
            self.population
            - self.epidemiological_data["Exposed"]
            - self.epidemiological_data["Active Cases"]
            - self.epidemiological_data["Total Recovered"]
            - self.epidemiological_data["Total Deaths"]
        )

        # Infected compartments.
        cdc_skew = (
            (
                self.epidemiological_data["Active Cases"]
                - self.epidemiological_data["inpatient_beds_used_covid"]
            )
            * (self.cases_by_vaccination["uv_mul"])
        ).astype(int)
        self.epidemiological_data["Infected_UV"] = cdc_skew

        cdc_skew = (
            (
                self.epidemiological_data["Active Cases"]
                - self.epidemiological_data["inpatient_beds_used_covid"]
            )
            * (self.cases_by_vaccination["fv_mul"])
        ).astype(int)
        self.epidemiological_data["Infected_FV"] = cdc_skew

        cdc_skew = (
            (
                self.epidemiological_data["Active Cases"]
                - self.epidemiological_data["inpatient_beds_used_covid"]
            )
            * (self.cases_by_vaccination["b_mul"])
        ).astype(int)
        self.epidemiological_data["Infected_BV"] = cdc_skew

        # Hospitalized compartments.
        cdc_skew = (
            self.epidemiological_data["inpatient_beds_used_covid"]
            * self.hospitalizations_by_vaccination["uv_mul"]
        ).astype(int)
        self.epidemiological_data["Hospitalized_UV"] = cdc_skew

        cdc_skew = (
            self.epidemiological_data["inpatient_beds_used_covid"]
            * self.hospitalizations_by_vaccination["fv_mul"]
        ).astype(int)
        self.epidemiological_data["Hospitalized_FV"] = cdc_skew

        cdc_skew = (
            self.epidemiological_data["inpatient_beds_used_covid"]
            * self.hospitalizations_by_vaccination["b_mul"]
        ).astype(int)
        self.epidemiological_data["Hospitalized_BV"] = cdc_skew

        # Recovered compartments.
        initial_recovered_unvaccinated_skew = (
            self.epidemiological_data["Total Recovered"].iloc[0]
            * self.cases_by_vaccination["uv_mul"].iloc[0]
        ).astype(int)

        initial_recovered_fully_vaccinated_skew = (
            self.epidemiological_data["Total Recovered"].iloc[0]
            * self.cases_by_vaccination["fv_mul"].iloc[0]
        ).astype(int)

        initial_recovered_booster_vaccinated_skew = (
            self.epidemiological_data["Total Recovered"].iloc[0]
            * self.cases_by_vaccination["b_mul"].iloc[0]
        ).astype(int)

        uv_to_fv = self.epidemiological_data[
            "percentage_unvaccinated_to_fully_vaccinated"
        ]
        fv_to_bv = self.epidemiological_data["percentage_fully_vaccinated_to_boosted"]

        self.epidemiological_data[["Recovered_UV", "Recovered_FV", "Recovered_BV"]] = 0
        self.epidemiological_data["Recovered_UV"].iloc[
            0
        ] = initial_recovered_unvaccinated_skew
        self.epidemiological_data["Recovered_FV"].iloc[
            0
        ] = initial_recovered_fully_vaccinated_skew
        self.epidemiological_data["Recovered_BV"].iloc[
            0
        ] = initial_recovered_booster_vaccinated_skew

        for i in range(1, len(self.epidemiological_data)):
            self.epidemiological_data["Recovered_UV"].iloc[i] = (
                self.epidemiological_data["Recovered_UV"].iloc[i - 1]
                + self.epidemiological_data["New Recoveries"].iloc[i]
                * self.cases_by_vaccination["uv_mul"].iloc[i]
                - uv_to_fv[i] * self.epidemiological_data["Recovered_UV"].iloc[i - 1]
            ).astype(int)
            self.epidemiological_data["Recovered_FV"].iloc[i] = (
                self.epidemiological_data["Recovered_FV"].iloc[i - 1]
                + self.epidemiological_data["New Recoveries"].iloc[i]
                * self.cases_by_vaccination["fv_mul"].iloc[i]
                + uv_to_fv[i] * self.epidemiological_data["Recovered_UV"].iloc[i - 1]
                - fv_to_bv[i] * self.epidemiological_data["Recovered_FV"].iloc[i - 1]
            ).astype(int)
            self.epidemiological_data["Recovered_BV"].iloc[i] = (
                self.epidemiological_data["Recovered_BV"].iloc[i - 1]
                + self.epidemiological_data["New Recoveries"].iloc[i]
                * self.cases_by_vaccination["b_mul"].iloc[i]
                + fv_to_bv[i] * self.epidemiological_data["Recovered_FV"].iloc[i - 1]
            ).astype(int)

        # Deceased compartments.
        initial_deceased_skew = (
            self.epidemiological_data["Total Deaths"].iloc[0]
            * self.cases_by_vaccination["uv_mul"].iloc[0]
        ).astype(int)
        cdc_skew = (
            self.epidemiological_data["New Deaths"]
            * self.cases_by_vaccination["uv_mul"]
        ).astype(int)
        cdc_skew[0] = 0
        cdc_skew = cdc_skew.cumsum()
        cdc_skew = cdc_skew + initial_deceased_skew
        self.epidemiological_data["Deceased_UV"] = cdc_skew

        initial_deceased_skew = (
            self.epidemiological_data["Total Deaths"].iloc[0]
            * self.cases_by_vaccination["fv_mul"].iloc[0]
        ).astype(int)
        cdc_skew = (
            self.epidemiological_data["New Deaths"]
            * self.cases_by_vaccination["fv_mul"]
        ).astype(int)
        cdc_skew[0] = 0
        cdc_skew = cdc_skew.cumsum()
        cdc_skew = cdc_skew + initial_deceased_skew
        self.epidemiological_data["Deceased_FV"] = cdc_skew

        initial_deceased_skew = (
            self.epidemiological_data["Total Deaths"].iloc[0]
            * self.cases_by_vaccination["b_mul"].iloc[0]
        ).astype(int)
        cdc_skew = (
            self.epidemiological_data["New Deaths"] * self.cases_by_vaccination["b_mul"]
        ).astype(int)
        cdc_skew[0] = 0
        cdc_skew = cdc_skew.cumsum()
        cdc_skew = cdc_skew + initial_deceased_skew
        self.epidemiological_data["Deceased_BV"] = cdc_skew

        # Accounting for "missing individuals".
        missing_individuals_unvaccinated = self.epidemiological_data[
            "unvaccinated_individuals"
        ] - (
            self.epidemiological_data["Infected_UV"]
            + self.epidemiological_data["Hospitalized_UV"]
            + self.epidemiological_data["Recovered_UV"]
            + self.epidemiological_data["Deceased_UV"]
        )

        missing_individuals_fully_vaccinated = self.epidemiological_data[
            "fully_vaccinated_individuals"
        ] - (
            self.epidemiological_data["Infected_FV"]
            + self.epidemiological_data["Hospitalized_FV"]
            + self.epidemiological_data["Recovered_FV"]
            + self.epidemiological_data["Deceased_FV"]
        )

        missing_individuals_booster_vaccinated = self.epidemiological_data[
            "boosted_individuals"
        ] - (
            self.epidemiological_data["Infected_BV"]
            + self.epidemiological_data["Hospitalized_BV"]
            + self.epidemiological_data["Recovered_BV"]
            + self.epidemiological_data["Deceased_BV"]
        )

        total_missing_individuals_vaccination = (
            missing_individuals_unvaccinated
            + missing_individuals_fully_vaccinated
            + missing_individuals_booster_vaccinated
        )

        self.epidemiological_data["Infected"] = (
            self.epidemiological_data["Infected_UV"]
            + self.epidemiological_data["Infected_FV"]
            + self.epidemiological_data["Infected_BV"]
        )

        self.epidemiological_data["Hospitalized"] = (
            self.epidemiological_data["Hospitalized_UV"]
            + self.epidemiological_data["Hospitalized_FV"]
            + self.epidemiological_data["Hospitalized_BV"]
        )

        self.epidemiological_data["Recovered"] = (
            self.epidemiological_data["Recovered_UV"]
            + self.epidemiological_data["Recovered_FV"]
            + self.epidemiological_data["Recovered_BV"]
        )

        self.epidemiological_data["Deceased"] = (
            self.epidemiological_data["Deceased_UV"]
            + self.epidemiological_data["Deceased_FV"]
            + self.epidemiological_data["Deceased_BV"]
        )

        # Adjusting Susceptible
        self.epidemiological_data["Susceptible_UV"] = (
            self.epidemiological_data["Susceptible"]
            * missing_individuals_unvaccinated
            / total_missing_individuals_vaccination
        ).astype(int)

        self.epidemiological_data["Susceptible_FV"] = (
            self.epidemiological_data["Susceptible"]
            * missing_individuals_fully_vaccinated
            / total_missing_individuals_vaccination
        ).astype(int)

        self.epidemiological_data["Susceptible_BV"] = (
            self.epidemiological_data["Susceptible"]
            * missing_individuals_booster_vaccinated
            / total_missing_individuals_vaccination
        ).astype(int)

        # Adjusting Exposed
        self.epidemiological_data["Exposed_UV"] = (
            self.epidemiological_data["Exposed"]
            * missing_individuals_unvaccinated
            / total_missing_individuals_vaccination
        ).astype(int)

        self.epidemiological_data["Exposed_FV"] = (
            self.epidemiological_data["Exposed"]
            * missing_individuals_fully_vaccinated
            / total_missing_individuals_vaccination
        ).astype(int)

        self.epidemiological_data["Exposed_BV"] = (
            self.epidemiological_data["Exposed"]
            * missing_individuals_booster_vaccinated
            / total_missing_individuals_vaccination
        ).astype(int)

        # Computing the total by vaccination statues across the different compartments.
        self.epidemiological_data["unvaccinated_compartment_total"] = (
            self.epidemiological_data["Susceptible_UV"]
            + self.epidemiological_data["Exposed_UV"]
            + self.epidemiological_data["Infected_UV"]
            + self.epidemiological_data["Hospitalized_UV"]
            + self.epidemiological_data["Recovered_UV"]
            + self.epidemiological_data["Deceased_UV"]
        )

        self.epidemiological_data["fully_vaccinated_compartment_total"] = (
            self.epidemiological_data["Susceptible_FV"]
            + self.epidemiological_data["Exposed_FV"]
            + self.epidemiological_data["Infected_FV"]
            + self.epidemiological_data["Hospitalized_FV"]
            + self.epidemiological_data["Recovered_FV"]
            + self.epidemiological_data["Deceased_FV"]
        )

        self.epidemiological_data["booster_vaccinated_compartment_total"] = (
            self.epidemiological_data["Susceptible_BV"]
            + self.epidemiological_data["Exposed_BV"]
            + self.epidemiological_data["Infected_BV"]
            + self.epidemiological_data["Hospitalized_BV"]
            + self.epidemiological_data["Recovered_BV"]
            + self.epidemiological_data["Deceased_BV"]
        )

        self.epidemiological_data["Original Infected"] = (
            self.epidemiological_data["Active Cases"]
            - self.epidemiological_data["inpatient_beds_used_covid"]
        )

        # Saving the epidemiological model data.
        self.epidemiological_data.iloc[:395].to_csv(
            f"{data_directory}/updated_data/epidemiological_model_data/pennsylvania.csv",
            index=False,
            columns=[
                "date",
                "unvaccinated_individuals",
                "fully_vaccinated_individuals",
                "boosted_individuals",
                "unvaccinated_compartment_total",
                "fully_vaccinated_compartment_total",
                "booster_vaccinated_compartment_total",
                "percentage_unvaccinated_to_fully_vaccinated",
                "percentage_fully_vaccinated_to_boosted",
                "New Cases",
                "Susceptible",
                "Exposed",
                "Infected",
                "Hospitalized",
                "Recovered",
                "Deceased",
                "Original Infected",
                "inpatient_beds_used_covid",
                "Total Recovered",
                "Total Deaths",
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
        )


epidemiological_data_preprocessing = EpidemiologicalDataPreProcessing(
    filepath=f"{data_directory}/updated_data/processed_state_data/pennsylvania.csv"
)
epidemiological_data_preprocessing.data_preprocessing()
