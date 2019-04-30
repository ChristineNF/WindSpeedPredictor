import os
import sys
import pandas as pd
import numpy as np
import ephem
import math
import datetime
import torch

from sklearn import preprocessing


class WindData:
    def __init__(self, name):
        self.name = name.lower()
        self.simulations = {}
        self.measurements = {}

        self.target = 'WS-92'
        self.target_height = 92
        self.sim_heights = [50, 75, 100, 150, 200, 250, 500]
        self.idx_tgth_lwr = None    # Index of lower height
        self.idx_tgth_upr = None    # Index of upper height
        self.tgth_lwr = None    # Lower height
        self.tgth_upr = None    # Upper height

        self.features = ['WS-92']
        self.n_features = None
        self.categories = ['season', 'WD_seaside', 'WD_sectors', 'RMOL3', 'RMOL5', 'daytime']
        self.category_mapping = {'season': ['winter', 'spring', 'summer', 'autumn'],
                                 'WD_seaside': ['land', 'sea'],
                                 'WD_sectors': ['north', 'east', 'south', 'west'],
                                 'RMOL3': ['unstable3', 'neutral3', 'stable3'],
                                 'RMOL5': ['unstable5', 'weakly unstable5', 'neutral5', 'weakly stable5', 'stable5'],
                                 'daytime': ['day', 'night']}
        self.meas_header = ['Time', 'WD-87', 'WS-92']

        self.observation_period = (datetime.datetime(year=2012, month=1, day=1),
                                   datetime.datetime(year=2015, month=12, day=31))
        self.training_dates = (datetime.datetime(year=2014, month=1, day=1),
                               datetime.datetime(year=2014, month=12, day=31))
        self.validation_dates = (datetime.datetime(year=2013, month=1, day=1),
                                 datetime.datetime(year=2013, month=12, day=31))
        self.test_dates = (datetime.datetime(year=2015, month=1, day=1),
                           datetime.datetime(year=2015, month=12, day=31))
        self.season_dates = {'winter': np.arange(1, 4),
                             'spring': np.arange(4, 7),
                             'summer': np.arange(7, 10),
                             'autumn': np.arange(10, 13)}

        self.root_dir = os.path.dirname(__file__)
        self._create_file_structure()

    def _get_directory(self, subdir=None):
        if subdir is None:
            directory_template = '{root_dir}/../../data/{name}/'
            return directory_template.format(root_dir=self.root_dir, name=self.name)
        else:
            subdir_template = '{root_dir}/../../data/{name}/{subdir}/'
            return subdir_template.format(root_dir=self.root_dir, name=self.name, subdir=subdir)

    def _create_file_structure(self):
        if not os.path.exists(self._get_directory()):
            print(f'Creating "{self.name}" directory and subdirectories for you!')
            os.makedirs(self._get_directory())
            for subdir in ['raw_measurement_data', 'raw_simulation_data', 'cleaned_data', 'model_data', 'models',
                           'model_settings', 'feature_settings', 'results' 'split_data']:
                os.makedirs(self._get_directory(subdir=subdir))
            print('Please insert the raw data appropriately and start the script again.')
            sys.exit()

    @staticmethod
    def _set_timestamp_index(df, index):
        df[index] = pd.to_datetime(df[index])
        return df.set_index(index)

    def _create_df(self, file_path, file_type, index, cols=None, **kwargs):
        function_name = f'read_{file_type}'
        df = pd.DataFrame()
        for f in os.listdir(file_path):
            df = pd.concat([df, getattr(pd, function_name)(file_path + f, **kwargs)], ignore_index=True)
        if cols is not None:
            df.columns = cols
        df = self._set_timestamp_index(df, index)
        return df.sort_index()

    # ----------------------------------------------------------------- #
    #                           READING FILES                           #
    # ----------------------------------------------------------------- #

    def read_raw_simulation_data(self, file_type='csv', index='Time', **kwargs):
        file_path = self._get_directory(subdir='raw_simulation_data')
        df = self._create_df(file_path, file_type, index, **kwargs)
        self.simulations.update({f'df': df})
        self.simulations['raw'] = self.simulations['df'].copy()  # Save all data
        return df

    def read_raw_measurement_data(self, file_type='csv', cols=None, index='Time', **kwargs):
        if cols is None:
            cols = self.meas_header
        file_path = self._get_directory(subdir='raw_measurement_data')
        df = self._create_df(file_path, file_type, index, cols=cols, **kwargs)
        self.measurements.update({'df': df})
        self.measurements['raw'] = self.measurements['df'].copy()
        return df

    # ----------------------------------------------------------------- #
    #                        REDUCE DATA FRAMES                         #
    # ----------------------------------------------------------------- #

    @staticmethod
    def _get_time_conditions(df, period):
        in_period = (period[0] <= df.index) & (df.index <= period[1])
        aligned_samples = (df.index.minute == 0) | (df.index.minute == 30)
        return in_period & aligned_samples

    def pick_target(self, target=None):
        if target is None:
            target = self.target
        self.measurements['df'] = self.measurements['df'][[target]]

    # Only timestamps within observation period at simulated sample time will be considered. NaNs are removed.
    def clean_timestamps(self, period=None):
        if period is None:
            period = self.observation_period
        conditions_sim = self._get_time_conditions(self.simulations['df'], period)
        self.simulations['df'] = self.simulations['df'][conditions_sim]
        self.simulations['complete'] = self.simulations['df'].copy()    # considered timestamps incl. NaN

        conditions_meas = self._get_time_conditions(self.measurements['df'], period)
        self.measurements['df'] = self.measurements['df'][conditions_meas]
        self.measurements['complete'] = self.measurements['df'].copy()  # considered timestamps incl. NaN

    def remove_nan(self):
        self.measurements['df'] = self.measurements['df'][np.isfinite(self.measurements['df'][self.target])]
        nnidx = self.measurements['df'].index
        self.simulations['df'] = self.simulations['df'][self.simulations['df'].index.isin(nnidx)]
        self.simulations['removed'] = self.simulations['complete'][~self.simulations['complete'].index.isin(nnidx)]

    # ----------------------------------------------------------------- #
    #                  INTERPOLATION TO TARGET HEIGHT                   #
    # ----------------------------------------------------------------- #

    # Assumption: target height is neither below lowest nor above highest simulation height
    def _get_column_indices(self):
        for i in range(len(self.sim_heights)):
            if self.sim_heights[i] > self.target_height:
                self.idx_tgth_lwr = i - 1
                self.idx_tgth_upr = i
                self.tgth_lwr = self.sim_heights[self.idx_tgth_lwr]
                self.tgth_upr = self.sim_heights[self.idx_tgth_upr]
                break

    def _unwrap_angular_discontinuity(self, col1_name, col2_name):
        condition = abs(self.simulations['df'][col2_name] - self.simulations['df'][col1_name]) > 180
        cond12 = self.simulations['df'][col1_name] < self.simulations['df'][col2_name]
        cond21 = self.simulations['df'][col1_name] > self.simulations['df'][col2_name]

        self.simulations['df'][col1_name][condition & cond12] += 360
        self.simulations['df'][col2_name][condition & cond21] += 360

    def _wrap_angular_discontinuity(self, colnames):
        for col in colnames:
            self.simulations['df'][col] = self.simulations['df'][col] % 360

    # Assumption: Always degrees not radians
    def interpolate_to_target_height(self, feat_name='WS', is_wd=False):
        self._get_column_indices()

        colname = f'{feat_name}-{self.target_height}'
        name1 = f'{feat_name}-{self.tgth_lwr}'
        name2 = f'{feat_name}-{self.tgth_upr}'

        if is_wd:
            self._unwrap_angular_discontinuity(name1, name2)
        m = (self.simulations['df'][name2] - self.simulations['df'][name1]) / (self.tgth_upr - self.tgth_lwr)
        intp = m * (self.target_height - self.tgth_lwr) + self.simulations['df'][name1]
        self.simulations['df'][colname] = intp
        if is_wd:
            self._wrap_angular_discontinuity([name1, name2, colname])

    # ----------------------------------------------------------------- #
    #                           EDIT FEATURES                           #
    # ----------------------------------------------------------------- #

    def add_seasonal_information(self, df):
        df['season'] = ['winter' if x.month in self.season_dates['winter'] else
                        'spring' if x.month in self.season_dates['spring'] else
                        'summer' if x.month in self.season_dates['summer'] else 'autumn' for x in df.index]
        return df

    def add_ws_shear_as_feature(self):
        ws_upper = self.simulations['df'][f'WS-{self.tgth_upr}']
        ws_lower = self.simulations['df'][f'WS-{self.tgth_lwr}']
        self.simulations['df']['WS_shear'] = ws_upper/ws_lower

    def add_wd_shear_as_feature(self):
        name1 = f'WD-{self.tgth_lwr}'
        name2 = f'WD-{self.tgth_upr}'

        self._unwrap_angular_discontinuity(name1, name2)
        wd_upper = self.simulations['df'][name2]
        wd_lower = self.simulations['df'][name1]
        self.simulations['df']['WD_shear'] = wd_upper - wd_lower
        self._wrap_angular_discontinuity([name1, name2, 'WD_shear'])

    def add_wd_as_sin_cos_as_feature(self):
        self.simulations['df']['WD_SIN'] = self.simulations['df'][f'WD-92'].apply(lambda x: math.sin(x * math.pi/180))
        self.simulations['df']['WD_COS'] = self.simulations['df'][f'WD-92'].apply(lambda x: math.cos(x * math.pi/180))

    def add_wd_sectors_as_feature(self):
        self.simulations['df']['WD_seaside'] = ['land' if ((x >= 90) and (x < 270)) else 'sea'
                                                for x in self.simulations['df'][f'WD-{self.target_height}']]
        self.simulations['df']['WD_sectors'] = ['east' if ((x >= 45) and (x < 135)) else
                                                'south' if ((x >= 135) and (x < 225)) else
                                                'west' if ((x >= 225) and (x < 315)) else 'north'
                                                for x in self.simulations['df'][f'WD-{self.target_height}']]

    def add_temperature_difference(self):
        self.simulations['df']['T_diff'] = self.simulations['df']['T-92'] - self.simulations['df']['TSK']

    @staticmethod
    def define_daytime(timestamp):
        sun = ephem.Sun()
        observer = ephem.Observer()
        observer.lat, observer.lon, observer.elevation = '52.8482', '3.4357', 0
        observer.date = timestamp
        sun.compute(observer)
        current_sun_alt = sun.alt * 180 / math.pi
        if current_sun_alt < 16:
            return 'night'
        else:
            return 'day'

    def add_time_as_feature(self):
        self.simulations['df'] = self.simulations['df'].reset_index()
        self.simulations['df']['Timestamp'] = self.simulations['df']['Time'].apply(lambda x: x.timestamp())
        self.simulations['df']['daytime'] = self.simulations['df']['Time'].apply(lambda x:
                                                                                 self.define_daytime(x))
        self.simulations['df'] = self._set_timestamp_index(self.simulations['df'], 'Time')

    def add_classified_rmol_as_feature(self):
        self.simulations['df']['RMOL3'] = ['unstable3' if x < -0.002 else
                                           'stable3' if x > 0.002 else
                                           'neutral3' for x in self.simulations['df']['RMOL']]
        self.simulations['df']['RMOL5'] = ['unstable5' if x < -0.01 else
                                           'weakly unstable5' if ((x >= -0.01) and (x < -0.002)) else
                                           'weakly stable5' if ((x > 0.002) and (x <= 0.01)) else
                                           'stable5' if x > 0.01 else
                                           'neutral5' for x in self.simulations['df']['RMOL']]

    def select_features(self, df, feature_set, norm, ohe, le, data_sets=None):
        if data_sets is None:
            data_sets = ['Train', 'Val', 'Test']
        features = []
        for fs in feature_set:
            if ohe and fs in self.category_mapping.keys():
                features.extend(self.category_mapping[fs])
            elif not (ohe or le) and (fs == 'RMOL3' or fs == 'RMOL5'):
                features.append('RMOL')
            else:
                if norm and fs in self.categories:
                    if data_sets == ['Train']:
                        self.n_features -= 1
                else:
                    features.append(fs)

        for ds in data_sets:
            df[ds] = df[ds][features]
        return df

    # ----------------------------------------------------------------- #
    #                         MODEL PREPARATION                         #
    # ----------------------------------------------------------------- #

    def do_one_hot_encoding(self, columns):
        for col in columns:
            categorical = pd.get_dummies(self.simulations['df'][col], dtype='int')
            self.simulations['df'] = pd.concat([self.simulations['df'], categorical], axis=1, sort=False)
        self.simulations['df'] = self.simulations['df'].drop(columns, axis=1)
        self.simulations['clean'] = self.simulations['df'].copy()

    def do_label_encoding(self, columns):
        for col in columns:
            le = preprocessing.LabelEncoder()
            le.fit(self.simulations['df'][col])
            self.simulations['df'][col] = le.transform(self.simulations['df'][col])

    def normalize_data(self, ohe=False, le=False):
        if not (ohe or le):
            self.simulations['df'] = self.simulations['df'].drop(self.categories, axis=1)
        xdata = self.simulations['df'].values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(xdata)
        self.simulations['df'] = pd.DataFrame(x_scaled, index=self.simulations['clean'].index,
                                              columns=self.simulations['df'].columns)

    # ----------------------------------------------------------------- #
    #                            SPLIT DATA                             #
    # ----------------------------------------------------------------- #

    def _convert_to_torch(self, name):
        self.simulations[name] = torch.tensor(self.simulations[name].values, dtype=torch.float)
        self.measurements[name] = torch.tensor(self.measurements[name].values, dtype=torch.float)

    def split_data(self, name, dates, use_torch=False):
        in_period_sim = self._get_time_conditions(self.simulations['df'], dates)
        in_period_meas = self._get_time_conditions(self.measurements['df'], dates)
        self.simulations[name] = self.simulations['df'][in_period_sim]
        self.measurements[name] = self.measurements['df'][in_period_meas]
        if use_torch:
            self._convert_to_torch(name=name)

    # ----------------------------------------------------------------- #
    #                        SAVE AND RELOAD DATA                       #
    # ----------------------------------------------------------------- #

    def save(self, data, filename, subdir=None, filetype='csv', *, index=True, **kwargs):
        filepath = f'{self._get_directory(subdir=subdir)}{filename}.{filetype}'
        getattr(data, f'to_{filetype}')(filepath, index=index, **kwargs)

    def _save_model_data(self, norm, ohe, le):
        f_sim = 'newa'
        if norm:
            f_sim += '_normed'
        if ohe:
            f_sim += '_ohe'
        if le:
            f_sim += '_le'
        self.simulations['df'].to_csv(self._get_directory(subdir='model_data') + f_sim)

    # Only split data frames are stored. If torch tensors are wanted, set use_torch True.
    def _save_split_data(self, norm, ohe, le, save_measurements):
        for splt in ['Train', 'Val', 'Test']:
            f_sim = f'sim_{splt}'
            if norm:
                f_sim += '_normed'
            if ohe:
                f_sim += '_ohe'
            if le:
                f_sim += '_le'
            self.save(self.simulations[splt], filename=f_sim, subdir='split_data')
            if save_measurements:
                self.save(self.measurements[splt], filename=f'meas_{splt}', subdir='split_data')

    def reload_data(self, filename, subdir='cleaned_data', filetype='csv', **kwargs):
        filepath = f'{self._get_directory(subdir=subdir)}{filename}.{filetype}'
        function_name = f'read_{filetype}'
        df = getattr(pd, function_name)(filepath, index_col=0, **kwargs)
        df.index = pd.to_datetime(df.index)
        return df

    def load_split_data(self, feature_set, norm=False, ohe=False, le=False, use_torch=False):
        self.n_features = len(feature_set)
        for splt in ['Train', 'Val', 'Test']:
            f_sim = f'sim_{splt}'
            f_meas = f'meas_{splt}'
            if norm:
                f_sim += '_normed'
            if ohe:
                f_sim += '_ohe'
            if le:
                f_sim += '_le'
            self.simulations[splt] = self.reload_data(f_sim, subdir='split_data')
            self.select_features(self.simulations, norm=norm, ohe=ohe, le=le, feature_set=feature_set, data_sets=[splt])
            self.measurements[splt] = self.reload_data(f_meas, subdir='split_data')
            if use_torch:
                self._convert_to_torch(name=splt)

    # ----------------------------------------------------------------- #
    #                PREPROCESSING AND MODEL PREPARATION                #
    # ----------------------------------------------------------------- #

    def preprocessing(self):
        self.read_raw_simulation_data()
        self.read_raw_measurement_data()
        self.pick_target()
        self.interpolate_to_target_height('WS')
        self.interpolate_to_target_height('WD', is_wd=True)
        self.interpolate_to_target_height('T')
        self.add_seasonal_information(self.simulations['df'])
        self.add_ws_shear_as_feature()
        self.add_temperature_difference()
        self.add_wd_as_sin_cos_as_feature()
        self.add_wd_sectors_as_feature()
        self.add_classified_rmol_as_feature()
        self.add_time_as_feature()
        self.clean_timestamps()
        self.remove_nan()
        self.save(data=self.simulations['df'], filename='clean_newa', subdir='cleaned_data')
        self.save(data=self.measurements['df'], filename='clean_measurements', subdir='cleaned_data')

    def model_preparation(self, do_ohe=False, do_le=False, norm=False, reload_file=True, save_split=True,
                          save_measurements=False):
        if reload_file:
            self.simulations['df'] = self.reload_data(filename='clean_newa')
            self.simulations['clean'] = self.reload_data(filename='clean_newa')
            self.measurements['df'] = self.reload_data(filename='clean_measurements')
        if do_ohe:
            self.do_one_hot_encoding(self.categories)
        if do_le:
            self.do_label_encoding(self.categories)
        if norm:
            self.normalize_data(do_ohe, do_le)
        self._save_model_data(norm, do_ohe, do_le)
        self.split_data(name='Train', dates=self.training_dates)
        self.split_data(name='Val', dates=self.validation_dates)
        self.split_data(name='Test', dates=self.test_dates)
        if save_split:
            self._save_split_data(norm, do_ohe, do_le, save_measurements)

    # ----------------------------------------------------------------- #
    #                              SETTER                               #
    # ----------------------------------------------------------------- #

    def set_target(self, target):
        self.target = target

    def set_target_height(self, target_height):
        self.target_height = target_height

    def set_sim_heights(self, sim_heights):
        self.sim_heights = sim_heights

    def set_features(self, features):
        self.features = features

    def set_measurement_header(self, header):
        self.meas_header = header

    def set_observation_period(self, observation_period):
        self.observation_period = observation_period

    def set_training_dates(self, training_dates):
        self.training_dates = training_dates

    def set_test_dates(self, test_dates):
        self.test_dates = test_dates
