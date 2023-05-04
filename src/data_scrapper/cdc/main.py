import sys
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import re
from src.settings import data_directory
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import urllib.request


# noinspection DuplicatedCode
class CDCDataScrapper:
    """This class implements a data scrapper which collects COVID-19 data from Centers for Disease Control and
    Prevention."""

    def __init__(
        self,
        website_url="https://covid.cdc.gov/covid-data-tracker/",
        output_path=f"{data_directory}/cdc/",
    ):
        """This method initializes the required parameters.

        :param website_url: String - Hyperlink of CDC website's COVID-19 data tracker.
        :param output_path: String - Path of the directory in which to store the scrapped data.
        """

        self.website_url = website_url
        self.output_path = output_path

        # self.data_links = {}
        with open("data_links") as f:
            self.data_links = json.loads(f.read())

        # for key in self.data_links:
        #     print(key, ":", self.data_links[key])

        self.download_links = {}
        self.data = {}

    def collect_data_links(self):
        """This method collects the links for the data sources."""

        # Collecting the links for the primary data categories.
        website = requests.get(self.website_url).text
        soup = BeautifulSoup(website, "html.parser")

        data = soup.find_all("button", {"class": "parentNav d-flex indPrtButton"})

        for button in data:
            self.data_links[button.string.strip()] = {
                "Main Link": "https://covid.cdc.gov/covid-data-tracker/#"
                + (button["data-tabname"])
            }

        # Collecting the links for the secondary data categories.
        options = Options()
        options.add_argument("--headless")

        chrome_webdriver = webdriver.Chrome(options=options)

        for primary_data_category in self.data_links:
            chrome_webdriver.get(self.data_links[primary_data_category]["Main Link"])
            time.sleep(2)

            elements = chrome_webdriver.find_elements(By.CLASS_NAME, "sub-card")

            for element in elements:
                data_source_name = element.text.partition("\n")[0]
                element_source = element.get_attribute("innerHTML")
                try:
                    data_tabname = "".join(
                        element_source.split('data-tabname="')[1].split('"')[0]
                    )
                except IndexError:
                    # print("Primary Data Category:", primary_data_category)
                    # print("Secondary Data Category:", data_source_name)
                    # print("Element Source:", element_source)
                    continue

                self.data_links[primary_data_category][data_source_name] = {
                    "Main Link": "https://covid.cdc.gov/covid-data-tracker/#"
                    + data_tabname
                }

        # with open('data_links', 'w') as f:
        #     f.write(json.dumps(self.data_links))
        #
        # with open('data_links') as f:
        #     test_dict = json.loads(f.read())

        # print("This is the Dictionary:\n", self.data_links)
        #
        # print(
        #     "\n This is how each key in the first level of the dictionary looks like:\n"
        # )
        # for key in self.data_links:
        #     print(key, ":", self.data_links[key])

    def download_data(self):
        # Collect download links.
        options = Options()
        options.add_argument("--headless")

        preferences = {"download.default_directory": self.output_path}
        options.add_experimental_option("prefs", preferences)

        chrome_webdriver = webdriver.Chrome(options=options)
        chrome_webdriver.get("https://covid.cdc.gov/covid-data-tracker/#county-view")
        time.sleep(2)

        elements = chrome_webdriver.find_elements(By.ID, "county-view-data-link")
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

        print(
            "\nTest:\n",
            self.data_links["Cases, Deaths, & Testing"]["County Data & Trends"],
        )

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
                time.sleep(2)

                element = chrome_webdriver.find_element(By.CLASS_NAME, "btn-container")
                element.click()
                time.sleep(2)

                element_source = element.get_attribute("innerHTML")

                csv_download_link = "https://data.cdc.gov" + "".join(
                    element_source.split('href="')[1].split('" data-type="CSV"')[0]
                )
                urllib.request.urlretrieve(
                    csv_download_link,
                    f"{self.output_path}/{download_link.split(' Download Link')[0]}.csv",
                )

        return

    def download_data2(self):
        # Collect download links.
        options = Options()
        # options.add_argument("--headless")

        preferences = {
            "download.default_directory": self.output_path.replace("/", "\\")
        }
        options.add_experimental_option("prefs", preferences)

        chrome_webdriver = webdriver.Chrome(options=options)
        chrome_webdriver.get("https://covid.cdc.gov/covid-data-tracker/#cases")
        time.sleep(2)

        elements = chrome_webdriver.find_elements(By.ID, "viewHistoricLink")
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

        element = chrome_webdriver.find_element(By.ID, "btnUSTableExport")
        chrome_webdriver.execute_script("arguments[0].click();", element)

        self.download_wait(self.output_path)

        print("Test:\n", self.data_links["Cases, Deaths, & Testing"])

        # Download data.
        for download_link in self.data_links["Cases, Deaths, & Testing"][
            "Cases, Deaths, and Testing by State"
        ]:
            if "Download Link" in download_link:
                chrome_webdriver.get(
                    self.data_links["Cases, Deaths, & Testing"][
                        "Cases, Deaths, and Testing by State"
                    ][download_link]
                )
                time.sleep(2)

                element = chrome_webdriver.find_element(By.CLASS_NAME, "btn-container")
                element.click()
                time.sleep(2)

                element_source = element.get_attribute("innerHTML")

                csv_download_link = "https://data.cdc.gov" + "".join(
                    element_source.split('href="')[1].split('" data-type="CSV"')[0]
                )
                urllib.request.urlretrieve(
                    csv_download_link,
                    f"{self.output_path}/{download_link.split(' Download Link')[0]}.csv",
                )

        return

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

    def scrape_all_data(self):
        """This method scrapes all the COVID-19 data."""

        for state in self.data_links:
            state_data_website_url = requests.get(self.data_links[state]).text
            state_data_soup = BeautifulSoup(state_data_website_url, "html.parser")

            self.data = {}

            dataset = self.create_final_dataset()
            dataset.to_csv(f"{self.output_path}{state}.csv", index=False)

    def create_final_dataset(self):
        """This method creates the final dataset.

        :returns dataframe: Pandas DataFrame -"""

        dataframe = pd.DataFrame.from_dict(self.data)
        dataframe["Dates"] = pd.to_datetime(dataframe["Dates"])

        return dataframe


cases_and_outcomes_data_scrapper = CDCDataScrapper(
    website_url="https://covid.cdc.gov/covid-data-tracker/",
    output_path=f"{data_directory}/cdc/",
)
# cases_and_outcomes_data_scrapper.collect_data_links()
cases_and_outcomes_data_scrapper.download_data2()
