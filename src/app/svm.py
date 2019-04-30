from app import Model
import os
import pickle
from sklearn.svm import SVR


class Svm(Model):
    def __init__(self, name, feature_df, target_df, target='WS-92'):
        super().__init__(name, feature_df, target_df, target=target)
        self.model_type = 'svm'
        self.model = None

    # ----------------------------------------------------------------- #
    #                             TRAINING                              #
    # ----------------------------------------------------------------- #

    def train_feature_data(self):
        svr = SVR(kernel=self.settings['kernel'], C=100, gamma='auto', epsilon=.1)
        svr.fit(self.feature_df['Train'], self.target_df['Train'][self.target])
        self.model = svr

    # ----------------------------------------------------------------- #
    #                           PERFORM MODEL                           #
    # ----------------------------------------------------------------- #

    def perform_svm(self, model_name, features, kernel, plot=False, recalc=True):
        self.settings['Model Name'] = model_name
        self.features = features
        self.settings['Feature Set'] = features.set_id
        self.settings['kernel'] = kernel

        if not os.path.exists(self.get_model_path()) or recalc:
            # Train model
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

    def set_kernel(self, kernel):
        self.settings['kernel'] = kernel
