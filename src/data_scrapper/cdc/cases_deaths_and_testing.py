import os
import json
import sys

from src.settings import data_directory
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import urllib.request
from fake_useragent import UserAgent


# noinspection DuplicatedCode
class CasesDeathsAndTesting:
    """This class implements a data scrapper which collects COVID-19 data from Centers for Disease Control and
    Prevention in the 'Cases, Deaths & Testing' data category."""

    def __init__(
        self,
        output_path=f"{data_directory}/cdc/",
    ):
        """This method initializes the required parameters.

        :param output_path: String - Path of the directory in which to store the scrapped data.
        """

        self.output_path = output_path

        # self.data_links = {}
        with open("data_links") as f:
            self.data_links = json.loads(f.read())

        # for key in self.data_links:
        #     print(key, ":", self.data_links[key])
        # sys.exit()

        self.data = {}

    def scrape_all_data(self):
        """This method scrapes all the COVID-19 data in the Primary Category of Cases Deaths and Testing."""
        # self.download_county_data_and_trends()
        # self.download_cases_deaths_and_testing_by_state()
        self.download_us_and_state_trends()

    def download_county_data_and_trends(self):
        """This method downloads the 'County Data & Trends' datasets."""

        # Collect download links.
        options = Options()
        options.add_argument("--headless")

        preferences = {"download.default_directory": self.output_path}
        options.add_experimental_option("prefs", preferences)

        chrome_webdriver = webdriver.Chrome(options=options)
        chrome_webdriver.get("https://covid.cdc.gov/covid-data-tracker/#county-view")

        elements = WebDriverWait(chrome_webdriver, 10).until(
            EC.presence_of_all_elements_located((By.ID, "county-view-data-link"))
        )

        for element in elements:
            data_source_names = element.text.split("\n")

            element_source = element.get_attribute("innerHTML")

            hyperlinks = [
                element_source.split('href="')[i].split('"')[0]
                for i in range(1, len(element_source.split('href="')))
            ]

            for i in range(len(data_source_names)):
                self.data_links["Cases, Deaths, & Testing"]["County Data & Trends"][
                    data_source_names[i] + " Download Link"
                ] = hyperlinks[i]

        # Download data.
        for download_link in self.data_links["Cases, Deaths, & Testing"][
            "County Data & Trends"
        ]:
            if "Download Link" in download_link:
                chrome_webdriver.get(
                    self.data_links["Cases, Deaths, & Testing"]["County Data & Trends"][
                        download_link
                    ]
                )

                element = WebDriverWait(chrome_webdriver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "btn-container"))
                )
                element.click()

                element_source = element.get_attribute("innerHTML")

                csv_download_link = "https://data.cdc.gov" + "".join(
                    element_source.split('href="')[1].split('" data-type="CSV"')[0]
                )
                urllib.request.urlretrieve(
                    csv_download_link,
                    f"{self.output_path}/{download_link.split(' Download Link')[0]}.csv",
                )

    def download_cases_deaths_and_testing_by_state(self):
        """This method downloads the 'Cases, Deaths, and Testing by State' datasets."""

        # Collect download links.
        options = Options()
        # options.add_argument("--headless")

        preferences = {
            "download.default_directory": self.output_path.replace("/", "\\")
        }
        options.add_experimental_option("prefs", preferences)

        chrome_webdriver = webdriver.Chrome(options=options)
        chrome_webdriver.get("https://covid.cdc.gov/covid-data-tracker/#cases")

        elements = WebDriverWait(chrome_webdriver, 10).until(
            EC.presence_of_all_elements_located((By.ID, "viewHistoricLink"))
        )

        for element in elements:
            data_source_names = element.text.split("\n")

            element_source = element.get_attribute("outerHTML")

            hyperlinks = [
                element_source.split('href="')[i].split('"')[0]
                for i in range(1, len(element_source.split('href="')))
            ]

            for i in range(len(data_source_names)):
                self.data_links["Cases, Deaths, & Testing"][
                    "Cases, Deaths, and Testing by State"
                ][data_source_names[i] + " Download Link"] = hyperlinks[i]

        # Download data.
        element = WebDriverWait(chrome_webdriver, 10).until(
            EC.presence_of_element_located((By.ID, "btnUSTableExport"))
        )
        chrome_webdriver.execute_script("arguments[0].click();", element)

        self.download_wait(self.output_path)

        for download_link in self.data_links["Cases, Deaths, & Testing"][
            "Cases, Deaths, and Testing by State"
        ]:
            if "Download Link" in download_link:
                chrome_webdriver.get(
                    self.data_links["Cases, Deaths, & Testing"][
                        "Cases, Deaths, and Testing by State"
                    ][download_link]
                )

                element = WebDriverWait(chrome_webdriver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "btn-container"))
                )
                element.click()

                element_source = element.get_attribute("innerHTML")

                csv_download_link = "https://data.cdc.gov" + "".join(
                    element_source.split('href="')[1].split('" data-type="CSV"')[0]
                )
                urllib.request.urlretrieve(
                    csv_download_link,
                    f"{self.output_path}/{download_link.split(' Download Link')[0]}.csv",
                )

    def download_us_and_state_trends(self):
        """This method downloads the 'US and State Trends' datasets."""

        # Collect download links.
        options = Options()
        options.add_argument("--headless")

        preferences = {
            "download.default_directory": self.output_path.replace("/", "\\")
        }
        options.add_experimental_option("prefs", preferences)

        chrome_webdriver = webdriver.Chrome(options=options)
        chrome_webdriver.get("https://covid.cdc.gov/covid-data-tracker/#trends")

        # Download data.
        element = WebDriverWait(chrome_webdriver, 5).until(
            EC.presence_of_element_located((By.ID, "btnUSTrendsTableExport"))
        )
        chrome_webdriver.execute_script("arguments[0].click();", element)

        self.download_wait(self.output_path)

    @staticmethod
    def download_wait(download_path):
        """This method waits for the file download to be completed.

        :param download_path - String - Download path of the file."""

        download_wait = True
        while download_wait:
            time.sleep(1)
            download_wait = False
            for file_name in os.listdir(download_path):
                if file_name.endswith(".crdownload"):
                    download_wait = True


cases_and_outcomes_data_scrapper = CasesDeathsAndTesting(
    output_path=f"{data_directory}/cdc/",
)
cases_and_outcomes_data_scrapper.scrape_all_data()
