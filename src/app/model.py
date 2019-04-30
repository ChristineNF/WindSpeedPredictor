import matplotlib.pylab as plt
import os
import csv
import pandas as pd
import numpy as np
from app import ErrorScores


class Model:
    def __init__(self, name, feature_df, target_df, target='WS-92'):
        self.name = name.lower()    # Project name (Ijmuiden)
        self.feature_df = feature_df    # Data Frame with cleaned feature data (NEWA)
        self.target_df = target_df  # Data Frame with cleaned measurement data (Meaurements WS-92)
        self.target = target    # Name of the target ('WS-92')
        self.settings = {}  # Settings of the model (used feature set, hyper parameter)
        self.features = None    # Selected features (instance of FeatureSet)
        self.errors = {}    # Error scores for each performed model
        # Percentage error of annual mean wind speed of all model performances
        self.model_errors = {'EVS': [], 'MSE': [], 'MAE': [], 'R2': [], 'Annual Mean WS': [], 'Error WS [m/s]': [],
                             'Error WS [%]': []}
        self.model_type = None  # ML model type
        self.model = None   # Trained model

        self.root_dir = os.path.dirname(__file__)   # Current directory
        self.fieldnames = ['Model Name', 'Model Type', 'Feature Set ID', 'EVS', 'MSE', 'MAE', 'R2', 'Annual Mean WS',
                           'Error WS [m/s]', 'Error WS [%]']    # Header in results file

    # ----------------------------------------------------------------- #
    #                            FILE PATHS                             #
    # ----------------------------------------------------------------- #

    def get_model_path(self):
        model_path_template = '{root_dir}/../../data/{name}/models/{model_type}_{model_name}.sav'
        return model_path_template.format(root_dir=self.root_dir, name=self.name,
                                          model_type=self.model_type,
                                          model_name=self.settings['Model Name'])

    def get_settings_path(self, feat_settings=False):
        if feat_settings:
            settings_template = '{root_dir}/../../data/{name}/feature_settings/feature_settings.csv'
            return settings_template.format(root_dir=self.root_dir, name=self.name)
        else:
            settings_template = '{root_dir}/../../data/{name}/model_settings/{model_type}_settings.csv'
            return settings_template.format(root_dir=self.root_dir, name=self.name, model_type=self.model_type)

    def get_results_path(self, file='results'):
        results_template = '{root_dir}/../../data/{name}/results/{file}.csv'
        return results_template.format(root_dir=self.root_dir, name=self.name, file=file)

    # ----------------------------------------------------------------- #
    #                           SAVE AND LOAD                           #
    # ----------------------------------------------------------------- #

    def _save_results(self):
        results = self.errors[self.settings['Model Name']]
        results.update({'Model Name': self.settings['Model Name'], 'Feature Set ID': self.features.set_id,
                        'Model Type': self.model_type})
        results_path = self.get_results_path()
        file_exists = os.path.isfile(results_path)

        with open(results_path, mode='a') as results_file:
            results_writer = csv.DictWriter(results_file, delimiter=',', lineterminator='\n',
                                            fieldnames=self.fieldnames)
            if not file_exists:
                results_writer.writeheader()
            results_writer.writerow(results)

    def save_model_mean_errors(self):
        file_path = self.get_results_path(file='mean_errors')
        file_exists = os.path.isfile(file_path)
        errors = {'Feature Set ID': self.features.set_id, 'Model Type': self.model_type}
        for k in self.model_errors.keys():
            errors[k] = self.get_mean_error(k)

        with open(file_path, mode='a') as mean_error_file:
            results_writer = csv.DictWriter(mean_error_file, delimiter=',', lineterminator='\n',
                                            fieldnames=errors.keys())
            if not file_exists:
                results_writer.writeheader()
            results_writer.writerow(errors)

    def save_settings(self):
        settings_path = self.get_settings_path()
        file_exists = os.path.isfile(settings_path)
        with open(settings_path, mode='a') as settings_file:
            settings_writer = csv.DictWriter(settings_file, delimiter=',', lineterminator='\n',
                                             fieldnames=self.settings.keys())
            if not file_exists:
                settings_writer.writeheader()
            else:
                settings_writer.writerow(self.settings)

    def _load_settings(self):
        with open(self.get_settings_path(), newline='') as settings_file:
            settings_reader = csv.DictReader(settings_file)
            for row in settings_reader:
                if row['Model Name'] == self.settings['Model Name']:
                    self.settings = row

    # ----------------------------------------------------------------- #
    #                            PREDICTION                             #
    # ----------------------------------------------------------------- #

    def predict_target(self, split_set, is_nn=False, is_lstm=False):
        if is_nn:
            self.target_df[f'{split_set}_Pred'] = (self.model(self.feature_df[split_set])).detach().numpy()
        elif is_lstm:
            # reset hidden layer
            self.model.hidden = self.model.init_hidden(len(self.feature_df['Lstm_' + split_set]))
                                                       #+ self.settings['seq_len']/2)  # TODO n_samples statt batch_size?
            self.target_df[f'{split_set}_Pred'] = self.model(self.feature_df['Lstm_' + split_set].transpose(0, 1))
        else:
            self.target_df[f'{split_set}_Pred'] = self.model.predict(self.feature_df[split_set])

    # ----------------------------------------------------------------- #
    #                            EVALUATION                             #
    # ----------------------------------------------------------------- #

    def test_model(self, split_set='Test', is_nn=False, is_lstm=False):
        if is_nn:
            actual = self.target_df[split_set]
            prediction = self.target_df[split_set + '_Pred']
        elif is_lstm:
            actual = (self.target_df[split_set][self.target][self.settings['seq_len']:]).values
            prediction = (self.target_df[split_set + '_Pred'][-1, :, :].squeeze()).detach().numpy()

        else:
            actual = self.target_df[split_set][self.target]
            prediction = self.target_df[split_set + '_Pred']

        err = ErrorScores(model_name=self.settings['Model Name'])
        err.evaluate(actual, prediction, is_nn, is_lstm)

        self.errors[err.model_name] = err.scores

        for err_type in self.model_errors.keys():
            self.model_errors[err_type].append(err.scores[err_type])

    def get_mean_error(self, err_type):
        return round(np.mean(self.model_errors[err_type]), 4)

    def plot_prediction(self, name, model_name, is_lstm=False):
        if is_lstm:
            actual = (self.target_df['Test'][self.target][self.settings['seq_len']:]).values
            pred = (self.target_df['Test_Pred'][-1, :, :].squeeze()).detach().numpy()
            print(actual.shape, pred.shape)
        else:
            actual = self.target_df[name]
            pred = self.target_df[f'{name}_Pred']

        plt.figure(1)
        plt.scatter(actual, pred, alpha=0.1)
        plt.xlabel('Measurements')
        plt.ylabel('Prediction')
        plt.title(f'{model_name} - Prediction of Measurements with {self.model_type}')
        plt.show()

    # ----------------------------------------------------------------- #
    #                            COMPARISON                             #
    # ----------------------------------------------------------------- #

    def compare_models(self, show_mean_errors=False, file_name='results'):
        if show_mean_errors:
            results = pd.read_csv(self.get_results_path(file='mean_errors'))
        else:
            results = pd.read_csv(self.get_results_path(file=file_name))

        categories = np.unique(results['Feature Set ID'])
        colors = np.linspace(0, 1, len(categories))
        colordict = dict(zip(categories, colors))
        results["Color"] = results['Feature Set ID'].apply(lambda x: colordict[x])

        plt.figure(1, figsize=(20, 10))
        for i, err in enumerate(['EVS', 'MSE', 'MAE', 'R2', 'Annual Mean WS', 'Error WS [%]']):
            plt.subplot(2, 3, i + 1)
            plt.scatter(results['Model Type'], results[err], c=results['Color'])
            plt.ylabel(f'{err}')
            if show_mean_errors:
                plt.title(f'Mean {err} for different Models')
            else:
                plt.title(f'{err} for different Models')
        plt.show()

    def rank_performances(self):
        results = pd.read_csv(self.get_results_path())
        res_sorted = results.sort_values(by='Error WS [%]')
        res_sorted = res_sorted[['Model Name', 'Feature Set ID', 'Model Type', 'Error WS [%]']]
        res_sorted.to_csv(self.get_results_path(file='ranking'))
