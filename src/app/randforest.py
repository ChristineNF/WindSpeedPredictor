from app import Model
import matplotlib.pylab as plt
import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class RandForest(Model):
    def __init__(self, name, feature_df, target_df, target='WS-92'):
        super().__init__(name, feature_df, target_df, target=target)
        self.model_type = 'rFor'
        self.model = None

# ----------------------------------------------------------------- #
    #                             TRAINING                              #
    # ----------------------------------------------------------------- #

    def train_feature_data(self):
        rforest = RandomForestRegressor(n_estimators=self.settings['n_estimators'],
                                        criterion=self.settings['criterion'],
                                        max_depth=self.settings['max_depth'])
        rforest.fit(self.feature_df['Train'], np.ravel(self.target_df['Train']))
        self.model = rforest

    # ----------------------------------------------------------------- #
    #                          HYPER PARAMETER                          #
    # ----------------------------------------------------------------- #

    def get_error(self, name):
        self.predict_target(name)
        return mean_squared_error(self.target_df[name], self.target_df[f'{name}_Pred'])

    def find_best_n_estimators(self, nest_selection, plot_error=False):
        training_errors = []
        validation_errors = []

        for nest in nest_selection:
            # print(f'Calculating n_estimators = {nest} ...')
            self.settings['n_estimators'] = nest
            self.train_feature_data()

            training_errors.append(self.get_error('Train'))
            validation_errors.append(self.get_error('Val'))

        if plot_error:
            plt.figure(1, figsize=(20, 10))
            plt.plot(nest_selection, training_errors, label='Training')
            plt.plot(nest_selection, validation_errors, label='Validation')
            plt.legend()
            plt.xlabel('n_estimators')
            plt.ylabel('MSE')
            title_str = 'MSE for Training and Validation depending on different n_estimators ({mtype}, {mname}'
            plt.title(title_str.format(mtype=self.model_type), mname=self.settings['Model Name'])
            plt.show()

        self.settings['n_estimators'] = nest_selection[validation_errors.index(min(validation_errors))]

    def find_best_max_depth(self, md_selection, plot_error=False):
        training_errors = []
        validation_errors = []

        for md in md_selection:
            # print(f'Calculating max_depth = {md} ...')
            self.settings['max_depth'] = md
            self.train_feature_data()

            training_errors.append(self.get_error('Train'))
            validation_errors.append(self.get_error('Val'))

        if plot_error:
            plt.figure(1, figsize=(20, 10))
            plt.plot(md_selection, training_errors, label='Training')
            plt.plot(md_selection, validation_errors, label='Validation')
            plt.legend()
            plt.xlabel('max_depth')
            plt.ylabel('MSE')
            plt.title('MSE for Training and Validation depending on different max_depth')
            plt.show()

        self.settings['max_depth'] = md_selection[validation_errors.index(min(validation_errors))]

    # ----------------------------------------------------------------- #
    #                           PERFORM MODEL                           #
    # ----------------------------------------------------------------- #

    def perform_rforest(self, model_name, features, n_estimators=None, nest_selection=None, max_depth=None,
                        md_selection=None, criterion='mse', random_state=0, plot=False, recalc=True):
        self.settings['Model Name'] = model_name
        self.features = features
        self.settings['Feature Set'] = features.set_id
        self.settings['criterion'] = criterion
        self.settings['random_state'] = random_state
        self.settings['n_estimators'] = n_estimators
        self.settings['max_depth'] = max_depth

        if not os.path.exists(self.get_model_path()) or recalc:
            if nest_selection is not None:
                # Find best n_estimators
                if max_depth is None:
                    self.settings['max_depth'] = 5  # Initial value
                self.find_best_n_estimators(nest_selection)
            if md_selection is not None:
                # Find best max_depth
                self.find_best_max_depth(md_selection)
            # Train model with found max_depth
            self.train_feature_data()
            # Save model
            self.save_settings()
            pickle.dump(self.model, open(self.get_model_path(), 'wb'))
        else:
            self._load_settings()
            self.model = pickle.load(open(self.get_model_path(), 'rb'))

        # Prediction
        self.predict_target('Test')
        # Test model with test data
        self.test_model()
        # Save results
        self._save_results()

        if plot:
            self.plot_prediction('Test', model_name)

    # ----------------------------------------------------------------- #
    #                              SETTER                               #
    # ----------------------------------------------------------------- #

    def set_model_name(self, model_name):
        self.settings['Model Name'] = model_name

    def set_n_estimators(self, n_estimators):
        self.settings['n_estimators'] = n_estimators

    def set_nest_selection(self, nest_selection):
        self.settings['nest_selection'] = nest_selection

    def set_max_depth(self, max_depth):
        self.settings['max_depth'] = max_depth

    def set_md_selection(self, md_selection):
        self.settings['md_selection'] = md_selection

    def set_criterion(self, criterion):
        self.settings['criterion'] = criterion

    def set_random_state(self, random_state):
        self.settings['random_state'] = random_state
