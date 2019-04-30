from app import Model
from app import Lstm
import matplotlib.pylab as plt
import os
import pickle

import torch
import torch.nn as nn


class LstmNetwork(Model):
    def __init__(self, name, feature_df, target_df, target='WS-92'):
        super().__init__(name, feature_df, target_df, target=target)
        self.model_type = 'lstm'
        self.model = None
        self.d_out = 1
        self.n_feat = len(self.feature_df['Train'].columns)
        self.n_samples = None

        self.criterion = nn.MSELoss()
        self.optimizer = None

    # ----------------------------------------------------------------- #
    #                             TRAINING                              #
    # ----------------------------------------------------------------- #

    def train_feature_data(self, plot=False):
        self.model = Lstm(self.n_feat, self.settings['hidden_dim'], d_out=self.d_out)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings['learning_rate'],
                                          weight_decay=self.settings['weight_decay'])
        loss_hist = []
        for t in range(self.settings['n_epochs']):
            n_batches = int(self.feature_df['Lstm_Train'].shape[0] / self.settings['batch_size'])
            batch_loss = 0.0
            for batch in range(0, n_batches):

                # Step 1. Calculate Batch
                batch_x = self.feature_df['Lstm_Train'][batch * self.settings['batch_size']:
                                                        (batch + 1) * self.settings['batch_size'], :]

                # convert to: sequence x batch_size x n_features TODO kann weg?

                batch_x = batch_x.transpose(0, 1)

                batch_y = self.target_df['Lstm_Train'][batch * self.settings['batch_size']:
                                                       (batch + 1) * self.settings['batch_size']]

                # Step 2. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM, detaching it from its history on the last
                # instance.
                self.model.hidden = self.model.init_hidden(self.settings['batch_size'])

                # Step 3. Run our forward pass.
                output = self.model(batch_x)

                # Step 4. Calculate the error with the last output
                loss = self.criterion(output[-1, :, :], batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item() / n_batches

            if t % 1 == 0:
                #print(batch_loss)
                loss_hist.append(batch_loss)
        if plot:
            plt.plot(loss_hist)
            plt.xlabel('Iteration')
            plt.ylabel('MSE')
            plt.title('Error Loss')
            plt.show()

    # ----------------------------------------------------------------- #
    #                         MODEL PREPARATION                         #
    # ----------------------------------------------------------------- #

    def prepare_input_data(self, name):
        feats = []
        trgts = []

        for i in range(0,  len(self.feature_df[name]) - self.settings['seq_len'], 1):
            feats.append(self.feature_df[name][i:i + self.settings['seq_len']].values)
            trgts.append(self.target_df[name][self.target][i + self.settings['seq_len'] - 1])
        self.n_samples = len(feats)
        self.feature_df['Lstm_' + name] = torch.tensor(feats, dtype=torch.float)
        self.target_df['Lstm_' + name] = torch.tensor(trgts, dtype=torch.float)

    # ----------------------------------------------------------------- #
    #                           PERFORM MODEL                           #
    # ----------------------------------------------------------------- #

    def perform_lstm(self, model_name, features, seq_len=7, hidden_dim=24, learning_rate=0.001, n_epochs=1000,
                     batch_size=128, weight_decay=0.0, recalc=True, plot=False):
        self.settings['Model Name'] = model_name
        self.features = features
        self.settings['Feature Set'] = features.set_id
        self.settings['seq_len'] = seq_len
        self.settings['hidden_dim'] = hidden_dim
        self.settings['embedded_dim'] = self.n_feat
        self.settings['learning_rate'] = learning_rate
        self.settings['n_epochs'] = n_epochs
        self.settings['batch_size'] = batch_size
        self.settings['weight_decay'] = weight_decay

        if not os.path.exists(self.get_model_path()) or recalc:
            # Build input tensors
            self.prepare_input_data(name='Train')
            # Train model with
            self.train_feature_data()
            # Save model
            self.save_settings()
            pickle.dump(self.model, open(self.get_model_path(), 'wb'))
        else:
            self._load_settings()
            self.model = pickle.load(open(self.get_model_path(), 'rb'))
        # Build input tensor Test data
        self.prepare_input_data(name='Test')
        # Prediction
        self.predict_target('Train', is_lstm=True)
        # Prediction
        self.predict_target('Test', is_lstm=True)
        # Test model with training data
        self.test_model(split_set='Train', is_lstm=True)
        # Test model with test data
        self.test_model(is_lstm=True)
        # Save results
        self._save_results()

        if plot:
            self.plot_prediction('Test', model_name, is_lstm=True)

    # ----------------------------------------------------------------- #
    #                              SETTER                               #
    # ----------------------------------------------------------------- #

    def set_model_name(self, model_name):
        self.settings['Model Name'] = model_name

    def set_seq_len(self, seq_len):
        self.settings['seq_len'] = seq_len

    def set_hidden_dim(self, hidden_dim):
        self.settings['hidden_dim'] = hidden_dim

    def set_learning_rate(self, learning_rate):
        self.settings['learning_rate'] = learning_rate

    def set_n_epochs(self, n_epochs):
        self.settings['n_epochs'] = n_epochs

    def set_batch_size(self, batch_size):
        self.settings['batch_size'] = batch_size

    def set_weight_decay(self, weight_decay):
        self.settings['weight_decay'] = weight_decay
