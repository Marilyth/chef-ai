from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
from Models.Instructions.ModuleBase import DecoderOnlyBase


class RNNTorch(DecoderOnlyBase):
    """The RNN model. This model is a simple RNN with an embedding layer and a linear output layer.
    This is done using the pytorch RNN module, which is much faster than a custom implementation.
    """
    def __init__(self, state_size: int, embedding_dimension: int, vocabulary_size: int, layers: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.state_size = state_size
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size

        self.rnn = nn.RNN(embedding_dimension, state_size, layers, batch_first=True)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.output = nn.Linear(state_size, vocabulary_size)

    def forward(self, x: torch.Tensor, hidden_states = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the RNN.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T) where B is the batch size and T is the sequence length.
            hidden_states (torch.Tensor, optional): The hidden states of the RNN. Defaults to None.
        Returns:
            torch.Tensor: The output of the RNN consisting of the logits and the hidden states.
        """
        B, T = x.shape

        x = self.embedding(x)
        
        # Forward pass through the RNN. If hidden_states is None, the hidden states are initialized to 0. Otherwise, the hidden states are used.
        out, hidden_states = self.rnn(x, hidden_states) if hidden_states is not None else self.rnn(x)

        # The output of the RNN is transformed to the vocabulary size.
        outputs = self.output(out)

        return outputs, hidden_states
