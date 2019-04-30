import torch
import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, n_features, hidden_dim, d_out):
        super(Lstm, self).__init__()
        self.num_layers = 1
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.d_out = d_out

        self.hidden = self.init_hidden(1)   # has no real function

        self.lstm = nn.LSTM(self.n_features, self.hidden_dim, self.num_layers)
        self.out = nn.Linear(self.hidden_dim, self.d_out)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, cell_input):
        lstm_out, self.hidden = self.lstm(cell_input, self.hidden)
        score = self.out(lstm_out)
        return score
