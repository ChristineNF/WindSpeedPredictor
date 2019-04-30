from app import Model
import matplotlib.pylab as plt
import os
import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class DecTree(Model):
    def __init__(self, name, feature_df, target_df, target='WS-92'):
        super().__init__(name, feature_df, target_df, target=target)
        self.model_type = 'tree'
        self.model = None

    # ----------------------------------------------------------------- #
    #                             TRAINING                              #
    # ----------------------------------------------------------------- #

    def train_feature_data(self):
        dtree = DecisionTreeRegressor(max_depth=self.settings['max_depth'], criterion=self.settings['criterion'],
                                      random_state=self.settings['random_state'])
        dtree.fit(self.feature_df['Train'], self.target_df['Train'])
        self.model = dtree

    # ----------------------------------------------------------------- #
    #                          HYPER PARAMETER                          #
    # ----------------------------------------------------------------- #

    def get_error(self, name):
        self.predict_target(name)
        return mean_squared_error(self.target_df[name], self.target_df[f'{name}_Pred'])

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

    def perform_dtree(self, model_name, features, max_depth=None, md_selection=None, criterion='mse', random_state=0,
                      plot=False, recalc=True):
        self.settings['Model Name'] = model_name
        self.features = features
        self.settings['Feature Set'] = features.set_id
        self.settings['criterion'] = criterion
        self.settings['random_state'] = random_state
        self.settings['max_depth'] = max_depth

        if not os.path.exists(self.get_model_path()) or recalc:
            # Find best max_depth
            if md_selection is not None:
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

    def set_max_depth(self, max_depth):
        self.settings['max_depth'] = max_depth

    def set_md_selection(self, md_selection):
        self.settings['md_selection'] = md_selection

    def set_criterion(self, criterion):
        self.settings['criterion'] = criterion

    def set_random_state(self, random_state):
        self.settings['random_state'] = random_state
