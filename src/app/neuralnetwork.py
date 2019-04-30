from app import Model
import matplotlib.pylab as plt
import os
import pickle
import numpy as np

import torch
import torch.nn as nn


class NeuralNetwork(Model):
    def __init__(self, name, feature_df, target_df, target='WS-92'):
        super().__init__(name, feature_df, target_df, target=target)
        self.model_type = 'nn'
        self.model = None

        self.D_in = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

    # ----------------------------------------------------------------- #
    #                             TRAINING                              #
    # ----------------------------------------------------------------- #

    def train_feature_data(self, plot=False):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings['learning_rate'])
        loss_hist = []
        for t in range(self.settings['n_iter']):
            # Forward step
            outputs = self.model(self.feature_df['Train'])
            # Error
            loss = self.criterion(outputs, self.target_df['Train'])
            # Backward Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if t % 10 == 0:
                loss_hist.append(loss.item())

        if plot:
            plt.plot(np.arange(0, self.settings['n_iter'], 10), loss_hist)
            plt.xlabel('Iteration')
            plt.ylabel('MSE')
            plt.title('Error Loss')
            plt.show()

    # ----------------------------------------------------------------- #
    #                         NETWORK SETTINGS                          #
    # ----------------------------------------------------------------- #

    def initialize_network(self):
        h1 = int(self.settings['D_in'] / 2)
        self.model = torch.nn.Sequential(
            nn.Linear(self.settings['D_in'], h1),
            nn.ReLU(),
            nn.Linear(h1, 1)
        )
    # ----------------------------------------------------------------- #
    #                           PERFORM MODEL                           #
    # ----------------------------------------------------------------- #

    def perform_nn(self, model_name, features, d_in, n_iter=10000, learning_rate=0.001, plot=False, recalc=True):
        self.settings['Model Name'] = model_name
        self.features = features
        self.settings['Feature Set'] = features.set_id
        self.settings['D_in'] = d_in
        self.settings['n_iter'] = n_iter
        self.settings['learning_rate'] = learning_rate

        if not os.path.exists(self.get_model_path()) or recalc:
            # Initialize Model
            self.initialize_network()
            # Train model with
            self.train_feature_data()
            # Save model
            self.save_settings()
            pickle.dump(self.model, open(self.get_model_path(), 'wb'))
        else:
            self._load_settings()
            self.model = pickle.load(open(self.get_model_path(), 'rb'))

        # Prediction
        self.predict_target('Test', is_nn=True)
        # Test model with test data
        self.test_model(is_nn=True)
        # Save results
        self._save_results()

        if plot:
            self.plot_prediction('Test', model_name)

    # ----------------------------------------------------------------- #
    #                              SETTER                               #
    # ----------------------------------------------------------------- #

    def set_model_name(self, model_name):
        self.settings['Model Name'] = model_name

    def set_n_iter(self, n_iter):
        self.settings['n_iter'] = n_iter

    def set_learning_rate(self, learning_rate):
        self.settings['learning_rate'] = learning_rate
