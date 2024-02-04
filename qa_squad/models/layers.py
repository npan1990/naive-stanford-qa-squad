import torch

import torch.nn.functional as F
from torch import nn

"""
Layers taken from BiDAF Tutorials.
"""


class CharacterEmbeddingLayer(nn.Module):

    def __init__(self, char_vocab_dim, char_emb_dim, num_output_channels, kernel_size):
        # TODO: Use num output channels
        super().__init__()
        self.char_emb_dim = char_emb_dim
        self.char_embedding = nn.Embedding(char_vocab_dim, char_emb_dim, padding_idx=1)
        self.char_convolution = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x = [bs, seq_len, word_len]
        batch_size = x.shape[0]
        x = self.dropout(self.char_embedding(x))
        # x = [bs, seq_len, word_len, char_emb_dim]
        x = x.permute(0, 1, 3, 2)
        # x = [bs, seq_len, char_emb_dim, word_len]
        x = x.view(-1, self.char_emb_dim, x.shape[3])
        # x = [bs*seq_len, char_emb_dim, word_len]
        x = x.unsqueeze(1)
        # x = [bs*seq_len, 1, char_emb_dim, word_len]
        x = self.relu(self.char_convolution(x))
        # x = [bs*seq_len, out_channels, H_out, W_out]
        x = x.squeeze()
        # x = [bs*seq_len, out_channels, W_out]
        x = F.max_pool1d(x, x.shape[2]).squeeze()
        # x = [bs*seq_len, out_channels, 1] => [bs*seq_len, out_channels]
        x = x.view(batch_size, -1, x.shape[-1])
        # x = [bs, seq_len, out_channels]
        return x


class HighwayNetwork(nn.Module):

    def __init__(self, input_dim, num_layers=2):
        super().__init__()

        self.num_layers = num_layers

        self.flow_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            flow_value = F.relu(self.flow_layer[i](x))
            gate_value = torch.sigmoid(self.gate_layer[i](x))

            x = gate_value * flow_value + (1 - gate_value) * x

        return x


class ContextualEmbeddingLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.highway_net = HighwayNetwork(input_dim)

    def forward(self, x):
        # x = [bs, seq_len, input_dim] = [bs, seq_len, emb_dim*2]

        highway_out = self.highway_net(x)
        # highway_out = [bs, seq_len, input_dim]

        outputs, _ = self.lstm(highway_out)
        # outputs = [bs, seq_len, emb_dim*2]

        return outputs