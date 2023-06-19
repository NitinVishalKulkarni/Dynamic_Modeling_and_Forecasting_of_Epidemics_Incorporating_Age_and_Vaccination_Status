import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from src.settings import data_directory

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

# Generating a list of epidemiological compartments.
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

# Datasets for the actual values and model forecasts.
location = "Pennsylvania"
compartment = "Infected"
vaccination_group = "V"
age_group = "18-49"

# # Fits and Forecasts:
# actual_values = pd.read_csv(f"{data_directory}/epidemiological_model_data/{location}.csv")
# # usa_model_fits = pd.read_csv(f"{data_directory}/epidemiological_model_parameters/model_predictions/USA.csv")
model_fits = pd.read_csv(f"{data_directory}/epidemic_forecasts/model_fit_plots/{location}.csv")
# model_forecasts = pd.read_csv(f"{data_directory}/epidemic_forecasts/average/{location}.csv")
# model_forecasts_lower = pd.read_csv(f"{data_directory}/epidemic_forecasts/average_lower/{location}.csv")
# model_forecasts_upper = pd.read_csv(f"{data_directory}/epidemic_forecasts/average_upper/{location}.csv")

# actual__values = (np.asarray(actual_values[f'{compartment}_{age_group}_{vaccination_group}'].iloc[-365:]))
# fit__values = (np.asarray(model_fits[f'{compartment}_{age_group}_{vaccination_group}'].iloc[:-90]))
# forecasted__values = (np.asarray(model_forecasts[f'{compartment}_{age_group}_{vaccination_group}']))
# forecasted__values_lower = (np.asarray(model_forecasts_lower[f'{compartment}_{age_group}_{vaccination_group}']))
# forecasted__values_upper = (np.asarray(model_forecasts_upper[f'{compartment}_{age_group}_{vaccination_group}']))

# Scenario Assessment:
actual_values = pd.read_csv(f"{data_directory}/epidemiological_model_data/{location}.csv")
model_forecasts_sdm_mm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/sdm and mm/mean/{location}.csv")
model_forecasts_lower_sdm_mm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/sdm and mm/lower/{location}.csv")
model_forecasts_upper_sdm_mm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/sdm and mm/upper/{location}.csv")

model_forecasts_sdm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/sdm/mean/{location}.csv")
model_forecasts_lower_sdm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/sdm/lower/{location}.csv")
model_forecasts_upper_sdm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/sdm/upper/{location}.csv")

model_forecasts_mm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/mm/mean/{location}.csv")
model_forecasts_lower_mm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/mm/lower/{location}.csv")
model_forecasts_upper_mm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/mm/upper/{location}.csv")

model_forecasts_ld_mm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/ld and mm/mean/{location}.csv")
model_forecasts_lower_ld_mm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/ld and mm/lower/{location}.csv")
model_forecasts_upper_ld_mm = pd.read_csv(f"{data_directory}/epidemic_forecasts/scenario_assessment/{location}/ld and mm/upper/{location}.csv")

actual__values = (np.asarray(actual_values[f'{compartment}_{age_group}_{vaccination_group}'].iloc[-90:]))
forecasted__values_sdm_mm = (np.asarray(model_forecasts_sdm_mm[f'{compartment}_{age_group}_{vaccination_group}']))
forecasted__values_lower_sdm_mm = (np.asarray(model_forecasts_lower_sdm_mm[f'{compartment}_{age_group}_{vaccination_group}']))
forecasted__values_upper_sdm_mm = (np.asarray(model_forecasts_upper_sdm_mm[f'{compartment}_{age_group}_{vaccination_group}']))

forecasted__values_sdm = (np.asarray(model_forecasts_sdm[f'{compartment}_{age_group}_{vaccination_group}']))
forecasted__values_lower_sdm = (np.asarray(model_forecasts_lower_sdm[f'{compartment}_{age_group}_{vaccination_group}']))
forecasted__values_upper_sdm = (np.asarray(model_forecasts_upper_sdm[f'{compartment}_{age_group}_{vaccination_group}']))

forecasted__values_mm = (np.asarray(model_forecasts_mm[f'{compartment}_{age_group}_{vaccination_group}']))
forecasted__values_lower_mm = (np.asarray(model_forecasts_lower_mm[f'{compartment}_{age_group}_{vaccination_group}']))
forecasted__values_upper_mm = (np.asarray(model_forecasts_upper_mm[f'{compartment}_{age_group}_{vaccination_group}']))

forecasted__values_ld_mm = (np.asarray(model_forecasts_ld_mm[f'{compartment}_{age_group}_{vaccination_group}']))
forecasted__values_lower_ld_mm = (np.asarray(model_forecasts_lower_ld_mm[f'{compartment}_{age_group}_{vaccination_group}']))
forecasted__values_upper_ld_mm = (np.asarray(model_forecasts_upper_ld_mm[f'{compartment}_{age_group}_{vaccination_group}']))

# min_value = np.min([np.min(actual__values), np.min(fit__values), np.min(forecasted__values)]) * 0.65
min_value = 1150
# max_value = np.max([np.max(actual__values), np.max(fit__values), np.max(forecasted__values)]) * 1.01
# max_value = int(max_value)
# max_value = int(np.max(forecasted__values_upper) * 1.03)
max_value = int(np.max(actual__values) * 1.08)

# print(max_value, np.max(actual__values), np.max(fit__values), np.max(forecasted__values))

# dates_actual = ['04/04/2021', '10/01/2021', '04/01/2022', '10/01/2022', '12/31/2022', '04/01/2023']
# dates_actual = ['04/01/2022', '10/01/2022', '12/31/2022', '04/01/2023']
dates_actual = ['01/01/2023', '02/01/2023', '03/01/2023', '04/01/2023']
x_actual = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in dates_actual]
days_actual = mdates.drange(x_actual[0], x_actual[-1], dt.timedelta(days=1))

# dates_fit = ['04/04/2021', '10/01/2021', '04/01/2022', '10/01/2022', '01/01/2023']
dates_fit = ['04/04/2021', '10/01/2021', '04/01/2022', '10/01/2022', '01/01/2023']
x_fit = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in dates_fit]
days_fit = mdates.drange(x_fit[0], x_fit[-1], dt.timedelta(days=1))

dates_forecasts = ['01/01/2023', '04/01/2023']
x_forecasts = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in dates_forecasts]
days_forecasts = mdates.drange(x_forecasts[0], x_forecasts[-1], dt.timedelta(days=1))

# plot_dates = ['04/01/2022', '07/01/2022', '10/01/2022', '01/01/2023', '03/31/2023']
plot_dates = ['01/01/2023', '02/01/2023', '03/01/2023', '03/31/2023']
x_plot_dates = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in plot_dates]
plot_days = mdates.drange(x_plot_dates[0], x_plot_dates[-1], dt.timedelta(days=1))

# Plots
plt.figure(figsize=(15, 10))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=1, interval=1))

# MODEL FIT AND FORECASTS:
# plt.plot(days_actual, actual__values, linewidth=2, c='red', label='Actual Infections')
# plt.plot(days_fit, fit__values, '-.', linewidth=2, c='green', label='Model Infections')
# plt.plot(days_forecasts, forecasted__values, '--', linewidth=2, c='blue', label='Forecasted Infections')
# plt.fill_between(days_forecasts, forecasted__values_lower, forecasted__values_upper, alpha=0.2)

# plt.plot(days_actual, actual__values, linewidth=2, c='red', label='Actual Hospitalizations')
# plt.plot(days_fit, fit__values, '-.', linewidth=2, c='green', label='Model Hospitalizations')
# plt.plot(days_forecasts, forecasted__values, '--', linewidth=2, c='blue', label='Forecasted Hospitalizations')
# plt.fill_between(days_forecasts, forecasted__values_lower, forecasted__values_upper, alpha=0.2)

# plt.plot(days_actual, actual__values, linewidth=2, c='red', label='Actual Deaths')
# plt.plot(days_fit, fit__values, '-.', linewidth=2, c='green', label='Model Deaths')
# plt.plot(days_forecasts, forecasted__values, '--', linewidth=2, c='blue', label='Forecasted Deaths')
# plt.fill_between(days_forecasts, forecasted__values_lower, forecasted__values_upper, alpha=0.2)

# SCENARIO ASSESSMENTS:
plt.plot(days_actual, actual__values, linewidth=2, c='red', label='Actual Infections')
plt.plot(days_forecasts, forecasted__values_sdm_mm, '--', linewidth=2, c='blue', label='SDM & MM Infections')
plt.fill_between(days_forecasts, forecasted__values_lower_sdm_mm, forecasted__values_upper_sdm_mm, alpha=0.2)

plt.plot(days_forecasts, forecasted__values_sdm, '.--', linewidth=2, c='teal', label='SDM Infections')
plt.fill_between(days_forecasts, forecasted__values_lower_sdm, forecasted__values_upper_sdm, alpha=0.2)

plt.plot(days_forecasts, forecasted__values_mm, '.', linewidth=2, c='violet', label='MM Infections')
plt.fill_between(days_forecasts, forecasted__values_lower_mm, forecasted__values_upper_mm, alpha=0.2)

# plt.plot(days_forecasts, forecasted__values_ld_mm, '--', linewidth=2, c='yellow', label='LD & MM Infections')
# plt.fill_between(days_forecasts, forecasted__values_lower_ld_mm, forecasted__values_upper_ld_mm, alpha=0.2)

step_size = 3000
plt.xlabel('Date', fontsize=32)
plt.ylabel('Population', fontsize=32)
# plt.title(f'Model Fit and Forecasts', fontsize=42)
plt.title(f'Scenario Assessment', fontsize=42)
# plt.xticks([x_plot_dates[0], x_plot_dates[1], x_plot_dates[2], x_plot_dates[3], x_plot_dates[4]], fontsize=24)
plt.xticks([x_plot_dates[0], x_plot_dates[1], x_plot_dates[2], x_plot_dates[3]], fontsize=24)
plt.yticks(np.arange(step_size, max_value, step_size),
           [f"{int(i / 1)}" for i in range(step_size, max_value, step_size)], fontsize=24)
# plt.yticks(fontsize=24)
plt.legend(fontsize=24, loc="best")
plt.ylim(ymin=min_value, ymax=max_value)
plt.xlim(xmin=x_actual[0], xmax=x_plot_dates[-1])
plt.grid()
plt.gcf().autofmt_xdate()
# plt.savefig('acutal_vs_awr_infections.png')
plt.show()


# test_dates = ['04/30/2021', '10/31/2021', '04/30/2022', '10/31/2022', '12/31/2022', '03/31/2023']
# test = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in test_dates]
#
# infected_actual = (np.asarray(ny_data['Infected']))
# infected_balanced = (np.asarray([balanced['Infected']]).ravel())
# infected_health = (np.asarray([health['Infected']]).ravel())
# infected_economy = (np.asarray([economy['Infected']]).ravel())
#
# min_value = min(np.min(infected_actual), np.min(infected_balanced), np.min(infected_health), np.min(infected_economy)) * 0.99
# max_value = max(np.max(infected_actual), np.max(infected_balanced), np.max(infected_health), np.max(infected_economy)) * 1.01
# step_size = 250_000
# max_value = int(max_value)

# # Plots
# plt.figure(figsize=(15, 10))
#
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# # plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=1, interval=1))
#
# plt.plot(days, infected_actual, linewidth=4, c='red', label='Actual Active Infections')
# plt.plot(days, infected_balanced, '--', linewidth=4, c='blue', label='Balanced Priority')
# plt.plot(days, infected_health, '.', linewidth=4, c='teal', label='Health Priority')
# plt.plot(days, infected_economy, '-.', linewidth=4, c='violet', label='EPP Priority')
#
# # plt.plot(days, hospitalized_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# # plt.plot(days, hospitalized_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')
#
# # plt.plot(days, deceased_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# # plt.plot(days, deceased_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')
#
# plt.xlabel('Date', fontsize=32)
# plt.ylabel('Population', fontsize=32)
# plt.title(f'Evolution of Infected Population', fontsize=42)
# plt.xticks([test[0], test[1], test[2], test[3], test[4], test[5]], fontsize=28)
# plt.yticks(np.arange(step_size, max_value, step_size),
#            [f"{float(i / 1000000)}M" for i in range(step_size, max_value, step_size)], fontsize=28)
# # plt.yticks(fontsize=20)
# plt.legend(fontsize=28)
# plt.ylim(ymin=min_value, ymax=max_value)
# plt.xlim(xmin=x[0], xmax=test[-1])
# plt.grid()
# plt.gcf().autofmt_xdate()
# # plt.savefig('acutal_vs_awr_infections.png')
# plt.show()



# deceased_actual = np.asarray(ny_data['Deceased'])
# deceased_balanced = np.asarray([balanced['Deceased']]).ravel()
# deceased_health = np.asarray([health['Deceased']]).ravel()
# deceased_economy = np.asarray([economy['Deceased']]).ravel()
# min_value = min(np.min(deceased_actual), np.min(deceased_balanced), np.min(deceased_health), np.min(deceased_economy)) * 0.99
# max_value = max(np.max(deceased_actual), np.max(deceased_balanced), np.max(deceased_health), np.max(deceased_economy)) * 1.01
# step_size = 10_000
# max_value = int(max_value)
# # Plots
# plt.figure(figsize=(15, 10))
#
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# # plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=1, interval=1))
#
# plt.plot(days, deceased_actual, linewidth=4, c='red', label='Actual Cumulative Deaths')
# plt.plot(days, deceased_balanced, '--', linewidth=4, c='blue', label='Balanced Priority')
# plt.plot(days, deceased_health, '.', linewidth=4, c='teal', label='Health Priority')
# plt.plot(days, deceased_economy, '-.', linewidth=4, c='violet', label='EPP Priority')
#
# # plt.plot(days, hospitalized_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# # plt.plot(days, hospitalized_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')
#
# # plt.plot(days, deceased_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# # plt.plot(days, deceased_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')
#
# plt.xlabel('Date', fontsize=32)
# plt.ylabel('Population', fontsize=32)
# plt.title(f'Evolution of Deceased Population', fontsize=42)
# plt.xticks([test[0], test[1], test[2], test[3], test[4], test[5]], fontsize=28)
# plt.yticks(np.arange(step_size, max_value, step_size),
#            [f"{int(i / 1000)}K" for i in range(step_size, max_value, step_size)], fontsize=28)
# # plt.yticks(fontsize=20)
# plt.legend(fontsize=28)
# plt.ylim(ymin=min_value, ymax=max_value)
# plt.xlim(xmin=x[0], xmax=test[-1])
# plt.grid()
# plt.gcf().autofmt_xdate()
# # plt.savefig('acutal_vs_awr_infections.png')
# plt.show()
#
#
#
# deceased_actual = np.cumsum(np.asarray(ny_data['Hospitalized']))
# deceased_balanced = np.cumsum(np.asarray([balanced['Hospitalized']]).ravel())
# deceased_health = np.cumsum(np.asarray([health['Hospitalized']]).ravel())
# deceased_economy = np.cumsum(np.asarray([economy['Hospitalized']]).ravel())
# min_value = min(np.min(deceased_actual), np.min(deceased_balanced), np.min(deceased_health), np.min(deceased_economy)) * 0.99
# max_value = max(np.max(deceased_actual), np.max(deceased_balanced), np.max(deceased_health), np.max(deceased_economy)) * 1.01
# step_size = 30_0000
# max_value = int(max_value)
# # Plots
# plt.figure(figsize=(15, 10))
#
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# # plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=1, interval=1))
#
# plt.plot(days, deceased_actual, linewidth=4, c='red', label='Actual Cumulative Hospitalizations')
# plt.plot(days, deceased_balanced, '--', linewidth=4, c='blue', label='Balanced Priority')
# plt.plot(days, deceased_health, '.', linewidth=4, c='teal', label='Health Priority')
# plt.plot(days, deceased_economy, '-.', linewidth=4, c='violet', label='EPP Priority')
#
# # plt.plot(days, hospitalized_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# # plt.plot(days, hospitalized_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')
#
# # plt.plot(days, deceased_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# # plt.plot(days, deceased_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')
#
# plt.xlabel('Date', fontsize=32)
# plt.ylabel('Population', fontsize=32)
# plt.title(f'Evolution of Hospitalized Population', fontsize=42)
# plt.xticks([test[0], test[1], test[2], test[3], test[4], test[5]], fontsize=28)
# plt.yticks(np.arange(step_size, max_value, step_size),
#            [f"{float(i / 1000)}K" for i in range(step_size, max_value, step_size)], fontsize=28)
# # plt.yticks(fontsize=20)
# plt.legend(fontsize=28)
# plt.ylim(ymin=min_value, ymax=max_value)
# plt.xlim(xmin=x[0], xmax=test[-1])
# plt.grid()
# plt.gcf().autofmt_xdate()
# # plt.savefig('acutal_vs_awr_infections.png')
# plt.show()



# # Infected Compartment
# step_size = 150_000
# # max_value = 740_000
# plt.figure(figsize=(20, 10))
# plt.plot(balanced['Infected'], label='AWR', linewidth=7)
# plt.plot(infected_actual, label='NY', linewidth=7)
# # plt.bar([i for i in range(len(balanced['Action'])) if balanced['Action'].iloc[i] == 4],
# #         [max_value for i in range(len(balanced['Action'])) if balanced['Action'].iloc[i] == 4],
# #         label='Lockdown + Mask Mandates', color='tab:orange', alpha=0.1)
# # plt.bar([i for i in range(len(balanced['Action'])) if balanced['Action'].iloc[i] == 2],
# #         [max_value for i in range(len(balanced['Action'])) if balanced['Action'].iloc[i] == 2],
# #         label='Mask Mandates', color='tab:blue', alpha=0.1)
# plt.legend(fontsize=28)
# plt.xlabel('Days', fontsize=32)
# plt.ylabel('Population', fontsize=32)
# plt.title('Infected Population Dynamics', fontsize=38)
# plt.grid()
# plt.xticks(np.arange(0, 181 + 1, 30), fontsize=30)
# plt.yticks(np.arange(step_size, max_value, step_size),
#            [f"{int((i / 1_000))}K" for i in range(step_size, max_value, step_size)], fontsize=30)
# plt.xlim([0, 181])
# plt.ylim(ymin=min_value, ymax=max_value)
# # plt.show()

# # Economic and Social Rate:
# plt.figure(figsize=(10, 10))
# plt.plot(balanced['Economic and Social Perception Rate'], color='blue', linestyle='--', linewidth=7, label='Balanced Priority')
# plt.plot(health['Economic and Social Perception Rate'], color='teal', linestyle='dotted', linewidth=7, label='Health Priority')
# plt.plot(economy['Economic and Social Perception Rate'], color='violet', linestyle='-.', linewidth=7, label='EPP Priority')
# plt.xlabel('Days', fontsize=32)
# plt.ylabel('Value', fontsize=32)
# plt.title('Economic & Public Perception Rate', fontsize=38)
# plt.grid()
# plt.legend(fontsize=24)
# plt.xticks(np.arange(0, 181 + 1, 30), fontsize=30)
# plt.yticks(np.arange(20, 101, 20), fontsize=30)
# plt.xlim([0, 181])
# plt.ylim([40, 101])
# plt.show()
