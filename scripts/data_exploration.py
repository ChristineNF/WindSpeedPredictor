from app import WindData
import pandas as pd
import numpy as np
import datetime
import matplotlib.pylab as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def detect_missing_data():
    msn_data = pd.DataFrame(columns=['month', 'N_msn'])
    nnull = []
    dates = []
    for yr in years:
        for m in range(1, 13):
            count = len(ij.measurements['complete'][[ij.target]][
                            (~np.isfinite(ij.measurements['complete'][ij.target])) & (
                                        ij.measurements['complete'].index.year == yr) & (
                                        ij.measurements['complete'].index.month == m)])
            msn_data = msn_data.append({'month': datetime.date(year=yr, month=m, day=1), 'N_msn': count},
                                       ignore_index=True)
            if count > 0:
                dates.append(datetime.date(year=yr, month=m, day=1))
                nnull.append(count)
    msn_data.set_index('month')

    plt.figure(2, figsize=(20, 10))
    plt.plot(msn_data['month'], msn_data['N_msn'], 'x')
    plt.grid(True)
    plt.title('Number of NaN per Month')


def show_wind_data_history():
    plt.figure(3, figsize=(10, 10))
    for i, yr in enumerate(years):
        plt.subplot(4, 1, i + 1)
        plt.plot(ij.simulations['df']['WS-92'][ij.simulations['df'].index.year == yr], alpha=0.5, label='NEWA')
        plt.plot(ij.measurements['df']['WS-92'][ij.measurements['df'].index.year == yr], alpha=0.5,
                 label='Measurements')
        plt.title(f'Wind Speeds {yr}')
        plt.ylabel('Wind Speed [m/s]')
        plt.ylim(-2, 40)
        plt.legend()


def show_frequency_distribution():
    plt.figure(6, figsize=(10, 35))
    plt.subplot(6, 2, 1)
    plt.hist([ij.simulations['df']['WS-92'], ij.measurements['df']['WS-92']], bins=np.arange(30),
             label=['NEWA', 'Measurements'])
    plt.legend()
    plt.title('Wind Speed Frequency Distribution 2012 - 2015')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Frequency')

    plt.subplot(6, 2, 2)
    plt.hist([ij.simulations['removed']['WS-92']], bins=np.arange(30))
    plt.title('Wind Speed Frequency Distribution of Removed NEWA Data due to Measurement Gaps')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Frequency')

    plt.subplot(6, 2, 3)
    plt.hist([ij.simulations['df']['WS-92'][ij.simulations['df'].index.year == yr] for yr in years],
             bins=np.arange(30), label=years)
    plt.legend()
    plt.title('Yearly Wind Speed Frequency Distribution NEWA')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Frequency')

    plt.subplot(6, 2, 4)
    plt.hist([ij.measurements['df']['WS-92'][ij.measurements['df'].index.year == yr] for yr in years],
             bins=np.arange(30), label=years)
    plt.legend()
    plt.title('Yearly Wind Speed Frequency Distribution Measurements')
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Frequency')

    for i, yr in enumerate(years):
        con1_sim = (ij.simulations['df'].index.year == yr)
        con1_meas = (ij.measurements['df'].index.year == yr)

        plt.subplot(6, 2, 5 + 2 * i)
        plt.hist([ij.simulations['df']['WS-92'][con1_sim & (ij.simulations['df']['season'] == sn)] for sn in seasons],
                 bins=np.arange(30), label=seasons)
        plt.legend()
        plt.title(f'Seasonal Wind Speed Frequency Distribution NEWA {yr}')
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Frequency')
        plt.ylim(0, 650)

        plt.subplot(6, 2, 5 + 2 * i + 1)
        plt.hist(
            [ij.measurements['df']['WS-92'][con1_meas & (ij.measurements['df']['season'] == sn)] for sn in seasons],
            bins=np.arange(30), label=seasons)
        plt.legend()
        plt.title(f'Seasonal Wind Speed Frequency Distribution Measurements {yr}')
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Frequency')
        plt.ylim(0, 650)


def show_scatter_of_all_data():
    xdata = ij.simulations['df']['WS-92']
    ydata = ij.measurements['df']['WS-92']
    mse = mean_squared_error(xdata.values, ydata.values)
    r2 = r2_score(xdata, ydata)

    plt.figure(7)
    plt.plot(xdata, ydata, '.', alpha=0.1)
    plt.xlabel('NEWA')
    plt.ylabel('Measurements')
    plt.title('NEWA vs. Measurements')
    plt.text(30, 35, f'MSE = {round(mse, 2)}')
    plt.text(30, 32, f'$R^2$ = {round(r2, 2)}')
    plt.xlim(0, 40)
    plt.ylim(0, 40)


def show_scatter_for_each_year():
    plt.figure(8, figsize=(12, 10))
    for i, yr in enumerate(years):
        xdata = ij.simulations['df']['WS-92'][ij.simulations['df'].index.year == yr]
        ydata = ij.measurements['df']['WS-92'][ij.measurements['df'].index.year == yr]
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


def show_scatter_for_each_season():
    plt.figure(9, figsize=(10, 35))
    for i, yr in enumerate(years):
        con1_sim = (ij.simulations['df'].index.year == yr)
        con1_meas = (ij.measurements['df'].index.year == yr)
        for j, sn in enumerate(seasons):
            xdata = ij.simulations['df']['WS-92'][con1_sim & (ij.simulations['df']['season'] == sn)]
            ydata = ij.measurements['df']['WS-92'][con1_meas & (ij.measurements['df']['season'] == sn)]
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


if __name__ == '__main__':
    # Creating WindData object
    ij = WindData('Ijmuiden')

    #####################################################################
    #                           PREPROCESSING                           #
    #####################################################################

    # Reading data
    ij.read_raw_simulation_data()
    ij.read_raw_measurement_data()
    # Reduce measurement data frame to target
    ij.pick_target(target='WS-92')
    # Add column with NEWA data interpolated to target height
    ij.interpolate_to_target_height('WS')
    ij.interpolate_to_target_height('WD', is_wd=True)
    ij.interpolate_to_target_height('T')
    # Add column with seasons
    ij.add_seasonal_information(ij.simulations['df'])
    ij.add_seasonal_information(ij.measurements['df'])
    # Remove date outside of observation period and congruent sampling frequency (30 min)
    ij.clean_timestamps(period=ij.observation_period)
    # Remove timestamps, where measurements are missing
    ij.remove_nan()

    #####################################################################
    #                         DATA EXPLORATION                          #
    #####################################################################

    years = [2012, 2013, 2014, 2015]
    seasons = ['winter', 'spring', 'summer', 'autumn']
    detect_missing_data()
    show_wind_data_history()
    show_frequency_distribution()
    show_scatter_of_all_data()
    show_scatter_for_each_year()
    show_scatter_for_each_season()
    plt.show()
