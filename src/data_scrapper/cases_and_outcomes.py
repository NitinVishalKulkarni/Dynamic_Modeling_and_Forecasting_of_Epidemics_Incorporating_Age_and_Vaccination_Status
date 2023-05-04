import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import re
from src.settings import data_directory


# noinspection DuplicatedCode
class CasesAndOutcomesDataScrapper:
    """This class implements a data scrapper which collects data on cases and outcomes from 'worldometers.info'."""

    def __init__(
        self,
        website_url="https://www.worldometers.info/coronavirus/country/us",
        output_path="../Data/Updated Data/cases_and_outcomes/",
    ):
        """This method initializes the required parameters.

        :param website_url: String - Hyperlink of worldometers website's COVID-19 page for the United States.
        :param output_path: String - Path of the directory in which to store the scrapped data.
        """

        self.website_url = website_url
        self.output_path = output_path
        self.data = {}

    def scrape_all_data(self):
        """This method scrapes all the COVID-19 data."""

        website = requests.get(self.website_url).text
        soup = BeautifulSoup(website, "html.parser")

        data = soup.find_all("a", {"class": "mt_a"})

        state_data_links = {}
        for a in data:
            state_data_links[a.string] = "https://www.worldometers.info" + (a["href"])

        for state in state_data_links:
            state_data_website_url = requests.get(state_data_links[state]).text
            state_data_soup = BeautifulSoup(state_data_website_url, "html.parser")

            self.data = {}
            self.scrape_total_cases_data(soup=state_data_soup)
            self.scrape_new_cases_data(soup=state_data_soup)
            self.scrape_active_cases_data(soup=state_data_soup)
            self.scrape_total_deaths_data(soup=state_data_soup)
            self.scrape_new_deaths_data(soup=state_data_soup)

            dataset = self.create_final_dataset()
            dataset.to_csv(f"{self.output_path}{state}.csv", index=False)

    def scrape_total_cases_data(self, soup):
        """This method scrapes the website for the data on total COVID-19 cases.

        :param soup - bs4.BeautifulSoup - Soup of the website we want to scrape data from.
        """

        data = soup.find(
            "script",
            type="text/javascript",
            string=re.compile("Highcharts.chart\\('coronavirus-cases-linear'"),
        ).text

        # Converting the extracted data to JSON format.
        data = re.sub(
            "Highcharts.chart\\('coronavirus-cases-linear', |Highcharts.chart\\('coronavirus-cases-log', ",
            "",
            data,
        )

        regular_expression = re.compile("(?<!\\\\)'")
        data = regular_expression.sub('"', data)

        data = re.sub(r"(\w+: )", r'"\1', data)
        data = re.sub(r"(: )", r'":', data)

        linear_data, logarithmic_data = data.split(");", 1)
        linear_data = re.sub("\\);", "", linear_data).strip()
        logarithmic_data = re.sub("\\);", "", logarithmic_data).strip()

        linear_data_json = json.loads(linear_data)
        logarithmic_data_json = json.loads(logarithmic_data)

        # Extracting the required data.
        dates = linear_data_json["xAxis"]["categories"]
        linear_data_values = linear_data_json["series"][0]["data"]

        logarithmic_data_values = logarithmic_data_json["series"][0]["data"]

        self.data["date"] = dates
        self.data["Total Cases (Linear)"] = linear_data_values
        self.data["Total Cases (Logarithmic)"] = logarithmic_data_values

    def scrape_new_cases_data(self, soup):
        """This method scrapes the website for the data on new COVID-19 cases.

        :param soup - bs4.BeautifulSoup - Soup of the website we want to scrape data from.
        """

        data = soup.find(
            "script",
            type="text/javascript",
            string=re.compile("Highcharts.chart\\('graph-cases-daily'"),
        ).text

        # Converting the extracted data to JSON format.
        data = re.sub(
            "Highcharts.chart\\('graph-cases-daily', |legendItemClick: function\\(event\\) {|"
            "this.checkbox.click\\(\\);|return false;\\s*}",
            "",
            data,
        )

        regular_expression = re.compile("(?<!\\\\)'")
        data = regular_expression.sub('"', data)

        data = re.sub(r"(\w+: )", r'"\1', data)
        data = re.sub(r"(: )", r'":', data)
        data = re.sub("\\);", "", data).strip()

        data = data.split(", function(chart)", 1)[0]

        data_json = json.loads(data)

        # Extracting the required data.
        daily_cases = data_json["series"][0]["data"]
        three_day_moving_average = data_json["series"][1]["data"]
        seven_day_moving_average = data_json["series"][2]["data"]

        self.data["Daily Cases"] = daily_cases
        self.data["Daily Cases (3-Day Moving Average)"] = three_day_moving_average
        self.data["Daily Cases (7-Day Moving Average)"] = seven_day_moving_average

    def scrape_active_cases_data(self, soup):
        """This method scrapes the website for the data on active COVID-19 cases.

        :param soup - bs4.BeautifulSoup - Soup of the website we want to scrape data from.
        """

        data = soup.find(
            "script",
            type="text/javascript",
            string=re.compile("Highcharts.chart\\('graph-active-cases-total'"),
        ).text

        # Converting the extracted data to JSON format.
        data = re.sub("Highcharts.chart\\('graph-active-cases-total',", "", data)

        regular_expression = re.compile("(?<!\\\\)'")
        data = regular_expression.sub('"', data)

        data = re.sub(r"(\w+: )", r'"\1', data)
        data = re.sub(r"(: )", r'":', data)
        data = re.sub("\\);", "", data).strip()

        data_json = json.loads(data)

        # Extracting the required data.
        active_cases = data_json["series"][0]["data"]

        self.data["Active Cases"] = active_cases

    def scrape_total_deaths_data(self, soup):
        """This method scrapes the website for the data on total COVID-19 deaths.

        :param soup - bs4.BeautifulSoup - Soup of the website we want to scrape data from.
        """

        data = soup.find(
            "script",
            type="text/javascript",
            string=re.compile("Highcharts.chart\\('coronavirus-deaths-linear'"),
        ).text

        # Converting the extracted data to JSON format.
        data = re.sub(
            "Highcharts.chart\\('coronavirus-deaths-linear', |Highcharts.chart\\('coronavirus-deaths-log', ",
            "",
            data,
        )

        regular_expression = re.compile("(?<!\\\\)'")
        data = regular_expression.sub('"', data)

        data = re.sub(r"(\w+: )", r'"\1', data)
        data = re.sub(r"(: )", r'":', data)

        linear_data, logarithmic_data = data.split(");", 1)
        linear_data = re.sub("\\);", "", linear_data).strip()
        logarithmic_data = re.sub("\\);", "", logarithmic_data).strip()

        linear_data_json = json.loads(linear_data)
        logarithmic_data_json = json.loads(logarithmic_data)

        # Extracting the required data.
        linear_data_values = linear_data_json["series"][0]["data"]

        logarithmic_data_values = logarithmic_data_json["series"][0]["data"]

        self.data["Total Deaths (Linear)"] = linear_data_values
        self.data["Total Deaths (Logarithmic)"] = logarithmic_data_values

    def scrape_new_deaths_data(self, soup):
        """This method scrapes the website for the data on new COVID-19 deaths.

        :param soup - bs4.BeautifulSoup - Soup of the website we want to scrape data from.
        """

        data = soup.find(
            "script",
            type="text/javascript",
            string=re.compile("Highcharts.chart\\('graph-deaths-daily'"),
        ).text

        # Converting the extracted data to JSON format.
        data = re.sub(
            "Highcharts.chart\\('graph-deaths-daily', |legendItemClick: function\\(event\\) {|"
            "this.checkbox.click\\(\\);|return false;\\s*}",
            "",
            data,
        )
        data = re.sub(r"visible: false,", r"visible: false", data)

        regular_expression = re.compile("(?<!\\\\)'")
        data = regular_expression.sub('"', data)

        data = re.sub(r"(\w+: )", r'"\1', data)
        data = re.sub(r"(: )", r'": ', data)
        data = re.sub("\\);", "", data).strip()

        data = data.split(", function(chart)", 1)[0]
        data_json = json.loads(data)

        # Extracting the required data.
        daily_deaths = data_json["series"][0]["data"]
        three_day_moving_average = data_json["series"][1]["data"]
        seven_day_moving_average = data_json["series"][2]["data"]

        self.data["Daily Deaths"] = daily_deaths
        self.data["Daily Deaths (3-Day Moving Average)"] = three_day_moving_average
        self.data["Daily Deaths (7-Day Moving Average)"] = seven_day_moving_average

    def create_final_dataset(self):
        """This method creates the final dataset.

        :returns dataframe: Pandas DataFrame -"""

        dataframe = pd.DataFrame.from_dict(self.data)
        dataframe["date"] = pd.to_datetime(dataframe["date"])

        return dataframe


cases_and_outcomes_data_scrapper = CasesAndOutcomesDataScrapper(
    website_url="https://www.worldometers.info/coronavirus/country/us",
    output_path=f"{data_directory}/cases_and_outcomes/",
)
cases_and_outcomes_data_scrapper.scrape_all_data()
