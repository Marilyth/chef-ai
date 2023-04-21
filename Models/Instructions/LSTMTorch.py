from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
from Models.Instructions.ModuleBase import DecoderOnlyBase


class LSTMTorch(DecoderOnlyBase):
    """The RNN model. This model is a simple RNN with an embedding layer and a linear output layer.
    This is done using the pytorch RNN module, which is much faster than a custom implementation.
    """
    def __init__(self, state_size: int, embedding_dimension: int, vocabulary_size: int, layers: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.state_size = state_size
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size

        self.lstm = nn.LSTM(embedding_dimension, state_size, layers, batch_first=True)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.output = nn.Linear(state_size, vocabulary_size)

    def forward(self, x: torch.Tensor, hidden_states = None, cell_states = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            hidden_states (torch.Tensor, optional): The hidden states. Defaults to None.
            cell_states (torch.Tensor, optional): The cell states. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The output, hidden states and cell states.
        """
        x = self.embedding(x)
        x, (hidden_states, cell_states) = self.lstm(x, (hidden_states, cell_states) if hidden_states is not None else None)
        x = self.output(x)

        return x, hidden_states, cell_states
