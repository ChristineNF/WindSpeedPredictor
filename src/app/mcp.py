from app import Model
from app import FeatureSet
import pandas as pd


class Mcp(Model):
    def __init__(self, name, feature_df, target_df, target='WS-92'):
        super().__init__(name, feature_df, target_df, target=target)
        self.model_type = 'mcp'

        self.mcp_training_df = pd.DataFrame({'Simulations': feature_df['Train'][target],
                                             'Measurements': target_df['Train'][target]})
        self.mcp_predict_df = pd.DataFrame({'Simulations': feature_df['Test'][target],
                                            'Measurements': target_df['Test'][target]})

        self.n_train = len(self.mcp_training_df.index)

    # ----------------------------------------------------------------- #
    #                          MODEL PARAMETER                          #
    # ----------------------------------------------------------------- #

    def _get_beta(self):
        covariance = sum(self.mcp_training_df['Sim x Meas']) - (sum(self.mcp_training_df['Simulations']) *
                                                                sum(self.mcp_training_df[
                                                                        'Measurements'])) / self.n_train
        variance = sum(self.mcp_training_df['Sim2']) - (sum(self.mcp_training_df['Simulations']) ** 2) / self.n_train
        return covariance / variance

    def _get_alpha(self, beta):
        return self.mcp_training_df['Measurements'].mean() - beta * self.mcp_training_df['Simulations'].mean()

    def _get_epsilon(self, alpha, beta):
        rand_train_sample = self.mcp_training_df.sample(1)
        return rand_train_sample['Measurements'].values - (beta * rand_train_sample['Simulations'].values + alpha)

    # ----------------------------------------------------------------- #
    #                           PREPARE DATA                            #
    # ----------------------------------------------------------------- #

    def _add_columns_to_mcp_training_df(self):
        self.mcp_training_df['Sim x Meas'] = self.mcp_training_df['Simulations'] * self.mcp_training_df['Measurements']
        self.mcp_training_df['Sim2'] = self.mcp_training_df['Simulations'] * self.mcp_training_df['Simulations']

    def _define_epsilon_for_test_set(self, alpha, beta):
        eps = []
        for i in range(len(self.mcp_predict_df)):
            eps.append(self._get_epsilon(alpha, beta)[0])
        self.mcp_predict_df['epsilon'] = eps

    # ----------------------------------------------------------------- #
    #                           PERFORM MODEL                           #
    # ----------------------------------------------------------------- #

    def perform_mcp(self, model_name, plot=False):
        self.settings['Model Name'] = model_name
        self.features = FeatureSet(0, [self.target])
        self.settings['Feature Set'] = self.features.set_id
        self._add_columns_to_mcp_training_df()
        beta = self._get_beta()
        alpha = self._get_alpha(beta)
        self._define_epsilon_for_test_set(alpha, beta)
        self.mcp_predict_df['Pred'] = beta * self.mcp_predict_df['Simulations'] + alpha + self.mcp_predict_df['epsilon']
        self.target_df['Test_Pred'] = self.mcp_predict_df['Pred'].copy()
        self.test_model()
        self._save_results()

        if plot:
            self.plot_prediction('Test', model_name)

    # ----------------------------------------------------------------- #
    #                              SETTER                               #
    # ----------------------------------------------------------------- #

    def set_model_name(self, model_name):
        self.settings['Model Name'] = model_name
