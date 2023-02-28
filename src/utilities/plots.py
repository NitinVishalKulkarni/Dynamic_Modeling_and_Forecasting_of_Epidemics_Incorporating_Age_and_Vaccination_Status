import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

balanced = pd.read_csv('./Results/Good Results/balanced.csv')
health = pd.read_csv('./Results/Good Results/health.csv')
economy = pd.read_csv('./Results/Good Results/economy.csv')
ny_data = pd.read_csv('epidemiological_model_data.csv').iloc[214:]

dates = ['11/01/2021', '12/01/2022', '01/01/2022', '02/01/2022', '03/01/2022', '05/01/2022']
x = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in dates]
days = mdates.drange(x[0], x[-1], dt.timedelta(days=1))

test_dates = ['11/30/2021', '12/31/2021', '01/31/2022', '02/28/2022', '03/31/2022', '04/30/2022']
test = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in test_dates]

infected_actual = (np.asarray(ny_data['Infected']))
infected_balanced = (np.asarray([balanced['Infected']]).ravel())
infected_health = (np.asarray([health['Infected']]).ravel())
infected_economy = (np.asarray([economy['Infected']]).ravel())

min_value = min(np.min(infected_actual), np.min(infected_balanced), np.min(infected_health), np.min(infected_economy)) * 0.99
max_value = max(np.max(infected_actual), np.max(infected_balanced), np.max(infected_health), np.max(infected_economy)) * 1.01
step_size = 250_000
max_value = int(max_value)

# Plots
plt.figure(figsize=(15, 10))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=1, interval=1))

plt.plot(days, infected_actual, linewidth=4, c='red', label='Actual Active Infections')
plt.plot(days, infected_balanced, '--', linewidth=4, c='blue', label='Balanced Priority')
plt.plot(days, infected_health, '.', linewidth=4, c='teal', label='Health Priority')
plt.plot(days, infected_economy, '-.', linewidth=4, c='violet', label='EPP Priority')

# plt.plot(days, hospitalized_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# plt.plot(days, hospitalized_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')

# plt.plot(days, deceased_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# plt.plot(days, deceased_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')

plt.xlabel('Date', fontsize=32)
plt.ylabel('Population', fontsize=32)
plt.title(f'Evolution of Infected Population', fontsize=42)
plt.xticks([test[0], test[1], test[2], test[3], test[4], test[5]], fontsize=28)
plt.yticks(np.arange(step_size, max_value, step_size),
           [f"{float(i / 1000000)}M" for i in range(step_size, max_value, step_size)], fontsize=28)
# plt.yticks(fontsize=20)
plt.legend(fontsize=28)
plt.ylim(ymin=min_value, ymax=max_value)
plt.xlim(xmin=x[0], xmax=test[-1])
plt.grid()
plt.gcf().autofmt_xdate()
# plt.savefig('acutal_vs_awr_infections.png')
plt.show()



deceased_actual = np.asarray(ny_data['Deceased'])
deceased_balanced = np.asarray([balanced['Deceased']]).ravel()
deceased_health = np.asarray([health['Deceased']]).ravel()
deceased_economy = np.asarray([economy['Deceased']]).ravel()
min_value = min(np.min(deceased_actual), np.min(deceased_balanced), np.min(deceased_health), np.min(deceased_economy)) * 0.99
max_value = max(np.max(deceased_actual), np.max(deceased_balanced), np.max(deceased_health), np.max(deceased_economy)) * 1.01
step_size = 10_000
max_value = int(max_value)
# Plots
plt.figure(figsize=(15, 10))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=1, interval=1))

plt.plot(days, deceased_actual, linewidth=4, c='red', label='Actual Cumulative Deaths')
plt.plot(days, deceased_balanced, '--', linewidth=4, c='blue', label='Balanced Priority')
plt.plot(days, deceased_health, '.', linewidth=4, c='teal', label='Health Priority')
plt.plot(days, deceased_economy, '-.', linewidth=4, c='violet', label='EPP Priority')

# plt.plot(days, hospitalized_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# plt.plot(days, hospitalized_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')

# plt.plot(days, deceased_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# plt.plot(days, deceased_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')

plt.xlabel('Date', fontsize=32)
plt.ylabel('Population', fontsize=32)
plt.title(f'Evolution of Deceased Population', fontsize=42)
plt.xticks([test[0], test[1], test[2], test[3], test[4], test[5]], fontsize=28)
plt.yticks(np.arange(step_size, max_value, step_size),
           [f"{int(i / 1000)}K" for i in range(step_size, max_value, step_size)], fontsize=28)
# plt.yticks(fontsize=20)
plt.legend(fontsize=28)
plt.ylim(ymin=min_value, ymax=max_value)
plt.xlim(xmin=x[0], xmax=test[-1])
plt.grid()
plt.gcf().autofmt_xdate()
# plt.savefig('acutal_vs_awr_infections.png')
plt.show()



deceased_actual = np.cumsum(np.asarray(ny_data['Hospitalized']))
deceased_balanced = np.cumsum(np.asarray([balanced['Hospitalized']]).ravel())
deceased_health = np.cumsum(np.asarray([health['Hospitalized']]).ravel())
deceased_economy = np.cumsum(np.asarray([economy['Hospitalized']]).ravel())
min_value = min(np.min(deceased_actual), np.min(deceased_balanced), np.min(deceased_health), np.min(deceased_economy)) * 0.99
max_value = max(np.max(deceased_actual), np.max(deceased_balanced), np.max(deceased_health), np.max(deceased_economy)) * 1.01
step_size = 30_0000
max_value = int(max_value)
# Plots
plt.figure(figsize=(15, 10))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(bymonthday=1, interval=1))

plt.plot(days, deceased_actual, linewidth=4, c='red', label='Actual Cumulative Hospitalizations')
plt.plot(days, deceased_balanced, '--', linewidth=4, c='blue', label='Balanced Priority')
plt.plot(days, deceased_health, '.', linewidth=4, c='teal', label='Health Priority')
plt.plot(days, deceased_economy, '-.', linewidth=4, c='violet', label='EPP Priority')

# plt.plot(days, hospitalized_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# plt.plot(days, hospitalized_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')

# plt.plot(days, deceased_actual, linewidth=2, c='red', label='Actual Hospitalizations')
# plt.plot(days, deceased_prediction, '--', linewidth=2, c='blue', label='Predicted Hospitalizations')

plt.xlabel('Date', fontsize=32)
plt.ylabel('Population', fontsize=32)
plt.title(f'Evolution of Hospitalized Population', fontsize=42)
plt.xticks([test[0], test[1], test[2], test[3], test[4], test[5]], fontsize=28)
plt.yticks(np.arange(step_size, max_value, step_size),
           [f"{float(i / 1000)}K" for i in range(step_size, max_value, step_size)], fontsize=28)
# plt.yticks(fontsize=20)
plt.legend(fontsize=28)
plt.ylim(ymin=min_value, ymax=max_value)
plt.xlim(xmin=x[0], xmax=test[-1])
plt.grid()
plt.gcf().autofmt_xdate()
# plt.savefig('acutal_vs_awr_infections.png')
plt.show()



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

# Economic and Social Rate:
plt.figure(figsize=(10, 10))
plt.plot(balanced['Economic and Social Perception Rate'], color='blue', linestyle='--', linewidth=7, label='Balanced Priority')
plt.plot(health['Economic and Social Perception Rate'], color='teal', linestyle='dotted', linewidth=7, label='Health Priority')
plt.plot(economy['Economic and Social Perception Rate'], color='violet', linestyle='-.', linewidth=7, label='EPP Priority')
plt.xlabel('Days', fontsize=32)
plt.ylabel('Value', fontsize=32)
plt.title('Economic & Public Perception Rate', fontsize=38)
plt.grid()
plt.legend(fontsize=24)
plt.xticks(np.arange(0, 181 + 1, 30), fontsize=30)
plt.yticks(np.arange(20, 101, 20), fontsize=30)
plt.xlim([0, 181])
plt.ylim([40, 101])
plt.show()
