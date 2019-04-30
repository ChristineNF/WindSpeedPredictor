from app import Model
import matplotlib.pylab as plt
import os
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


class Knn(Model):
    def __init__(self, name, feature_df, target_df, target='WS-92'):
        super().__init__(name, feature_df, target_df, target=target)
        self.model_type = 'knn'
        self.model = None

    # ----------------------------------------------------------------- #
    #                             TRAINING                              #
    # ----------------------------------------------------------------- #

    def train_feature_data(self):
        knn = KNeighborsRegressor(self.settings['k'], weights=self.settings['knn_weights'])
        knn.fit(self.feature_df['Train'], self.target_df['Train'])
        self.model = knn

    # ----------------------------------------------------------------- #
    #                          HYPER PARAMETER                          #
    # ----------------------------------------------------------------- #

    def get_error(self, name):
        self.predict_target(name)
        return mean_squared_error(self.target_df[name], self.target_df[f'{name}_Pred'])

    def find_best_k(self, k_selection, plot_error=False):
        training_errors = []
        validation_errors = []

        for k in k_selection:
            # print(f'Calculating k = {k} ...')
            self.settings['k'] = k
            self.train_feature_data()

            training_errors.append(self.get_error('Train'))
            validation_errors.append(self.get_error('Val'))

        if plot_error:
            plt.figure(1, figsize=(20, 10))
            plt.plot(k_selection, training_errors, label='Training')
            plt.plot(k_selection, validation_errors, label='Validation')
            plt.legend()
            plt.xlabel('k')
            plt.ylabel('MSE')
            plt.title('MSE for Training and Validation depending on different k')
            plt.show()

        self.settings['k'] = k_selection[validation_errors.index(min(validation_errors))]

    # ----------------------------------------------------------------- #
    #                           PERFORM MODEL                           #
    # ----------------------------------------------------------------- #

    def perform_knn(self, model_name, features, k_selection, knn_weights='uniform', plot=False, recalc=True):
        self.settings['Model Name'] = model_name
        self.features = features
        self.settings['Feature Set'] = features.set_id
        self.settings['knn_weights'] = knn_weights

        if not os.path.exists(self.get_model_path()) or recalc:
            # Find best k
            self.find_best_k(k_selection)
            # Train model with found k
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

    def set_knn_weights(self, knn_weights):
        self.settings['knn_weights'] = knn_weights

    def set_k(self, k):
        self.settings['k'] = k
