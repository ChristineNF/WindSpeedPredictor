from app import WindData
import pandas as pd
import numpy as np
import datetime
import matplotlib.pylab as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class DataExploration:
    def __init__(self, name):
        self.name = name.lower()
        self.winddata = WindData(self.name)
        self.winddata.preprocessing()
        self.winddata.add_seasonal_information(self.winddata.measurements['df'])

        self.years = [2012, 2013, 2014, 2015]
        self.seasons = ['winter', 'spring', 'summer', 'autumn']

    def detect_missing_data(self):
        msn_data = pd.DataFrame(columns=['month', 'N_msn'])
        nnull = []
        dates = []
        for yr in self.years:
            for m in range(1, 13):
                count = len(self.winddata.measurements['complete'][[self.winddata.target]][
                                (~np.isfinite(self.winddata.measurements['complete'][self.winddata.target])) & (
                                        self.winddata.measurements['complete'].index.year == yr) & (
                                        self.winddata.measurements['complete'].index.month == m)])
                msn_data = msn_data.append({'month': datetime.date(year=yr, month=m, day=1), 'N_msn': count},
                                           ignore_index=True)
                if count > 0:
                    dates.append(datetime.date(year=yr, month=m, day=1))
                    nnull.append(count)
        msn_data.set_index('month')

        plt.figure(1, figsize=(20, 10))
        plt.plot(msn_data['month'], msn_data['N_msn'], 'x')
        plt.grid(True)
        plt.title('Number of NaN per Month')
        plt.show()

    def show_wind_data_history(self):
        plt.figure(2, figsize=(20, 20))
        for i, yr in enumerate(self.years):
            plt.subplot(4, 1, i + 1)
            plt.plot(self.winddata.simulations['df']['WS-92'][self.winddata.simulations['df'].index.year == yr],
                     alpha=0.5, label='NEWA')
            plt.plot(self.winddata.measurements['df']['WS-92'][self.winddata.measurements['df'].index.year == yr],
                     alpha=0.5, label='Measurements')
            plt.title(f'Wind Speeds {yr}')
            plt.ylabel('Wind Speed [m/s]')
            plt.ylim(-2, 40)
            plt.legend()
        plt.show()

    def show_frequency_distribution(self):
        plt.figure(3, figsize=(20, 35))
        plt.subplot(6, 2, 1)
        plt.hist([self.winddata.simulations['df']['WS-92'], self.winddata.measurements['df']['WS-92']],
                 bins=np.arange(30), label=['NEWA', 'Measurements'])
        plt.legend()
        plt.title('Wind Speed Frequency Distribution 2012 - 2015')
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Frequency')

        plt.subplot(6, 2, 2)
        plt.hist([self.winddata.simulations['removed']['WS-92']], bins=np.arange(30))
        plt.title('Wind Speed Frequency Distribution of Removed NEWA Data due to Measurement Gaps')
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Frequency')

        plt.subplot(6, 2, 3)
        plt.hist([self.winddata.simulations['df']['WS-92'][self.winddata.simulations['df'].index.year == yr] for yr in
                  self.years], bins=np.arange(30), label=self.years)
        plt.legend()
        plt.title('Yearly Wind Speed Frequency Distribution NEWA')
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Frequency')

        plt.subplot(6, 2, 4)
        plt.hist([self.winddata.measurements['df']['WS-92'][self.winddata.measurements['df'].index.year == yr] for yr in
                  self.years], bins=np.arange(30), label=self.years)
        plt.legend()
        plt.title('Yearly Wind Speed Frequency Distribution Measurements')
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Frequency')

        for i, yr in enumerate(self.years):
            con1_sim = (self.winddata.simulations['df'].index.year == yr)
            con1_meas = (self.winddata.measurements['df'].index.year == yr)

            plt.subplot(6, 2, 5 + 2 * i)
            plt.hist([self.winddata.simulations['df']['WS-92'][con1_sim & (self.winddata.simulations['df']['season'] ==
                                                                           sn)] for sn in self.seasons],
                     bins=np.arange(30), label=self.seasons)
            plt.legend()
            plt.title(f'Seasonal Wind Speed Frequency Distribution NEWA {yr}')
            plt.xlabel('Wind Speed [m/s]')
            plt.ylabel('Frequency')
            plt.ylim(0, 650)

            plt.subplot(6, 2, 5 + 2 * i + 1)
            plt.hist(
                [self.winddata.measurements['df']['WS-92'][con1_meas & (self.winddata.measurements['df']['season'] ==
                                                                        sn)] for sn in self.seasons],
                bins=np.arange(30), label=self.seasons)
            plt.legend()
            plt.title(f'Seasonal Wind Speed Frequency Distribution Measurements {yr}')
            plt.xlabel('Wind Speed [m/s]')
            plt.ylabel('Frequency')
            plt.ylim(0, 650)
        plt.show()

    def show_scatter_of_all_data(self):
        xdata = self.winddata.simulations['df']['WS-92']
        ydata = self.winddata.measurements['df']['WS-92']
        mse = mean_squared_error(xdata.values, ydata.values)
        r2 = r2_score(xdata, ydata)

        plt.figure(4)
        plt.plot(xdata, ydata, '.', alpha=0.1)
        plt.xlabel('NEWA')
        plt.ylabel('Measurements')
        plt.title('NEWA vs. Measurements')
        plt.text(30, 35, f'MSE = {round(mse, 2)}')
        plt.text(30, 32, f'$R^2$ = {round(r2, 2)}')
        plt.xlim(0, 40)
        plt.ylim(0, 40)
        plt.show()

    def show_scatter_for_each_year(self):
        plt.figure(5, figsize=(12, 10))
        for i, yr in enumerate(self.years):
            xdata = self.winddata.simulations['df']['WS-92'][self.winddata.simulations['df'].index.year == yr]
            ydata = self.winddata.measurements['df']['WS-92'][self.winddata.measurements['df'].index.year == yr]
            mse = mean_squared_error(xdata, ydata)
            r2 = r2_score(xdata, ydata)

            plt.subplot(2, 2, i + 1)
            plt.scatter(xdata, ydata, alpha=0.1)
            plt.xlabel('NEWA')
            plt.ylabel('Measurements')
            plt.title(f'NEWA vs. Measurements {yr}')
            plt.text(30, 35, f'MSE = {round(mse, 2)}')
            plt.text(30, 32, f'$R^2$ = {round(r2, 2)}')
            plt.xlim(0, 40)
            plt.ylim(0, 40)
        plt.show()

    def show_scatter_for_each_season(self):
        plt.figure(6, figsize=(20, 35))
        for i, yr in enumerate(self.years):
            con1_sim = (self.winddata.simulations['df'].index.year == yr)
            con1_meas = (self.winddata.measurements['df'].index.year == yr)
            for j, sn in enumerate(self.seasons):
                xdata = self.winddata.simulations['df']['WS-92'][con1_sim & (self.winddata.simulations['df']['season']
                                                                             == sn)]
                ydata = self.winddata.measurements['df']['WS-92'][con1_meas & (
                        self.winddata.measurements['df']['season'] == sn)]
                mse = mean_squared_error(xdata, ydata)
                r2 = r2_score(xdata, ydata)

                plt.subplot(7, 4, 4 * i + j + 1)
                plt.scatter(xdata, ydata, label=sn, alpha=0.1)
                plt.xlabel('NEWA')
                plt.ylabel('Measurements')
                plt.title(f'NEWA vs. Measurements for {sn} {yr}')
                plt.text(27, 35, f'MSE = {round(mse, 2)}')
                plt.text(27, 32, f'$R^2$ = {round(r2, 2)}')
                plt.xlim(0, 40)
                plt.ylim(0, 40)
        plt.show()
