from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time


class GRUCell(nn.Module):
    def __init__(self, state_size: int, embedding_dimension: int):
        nn.GRU
        super().__init__()
        self.state_size = state_size
        self.embedding_dimension = embedding_dimension

        # Weights for the update gate, reset gate and hidden state.
        # The state size is doubled because the hidden state is concatenated with the input vector for efficiency.
        # Effectively this is the same as having two separate weight matrices for the input and hidden state.
        self.W_z = nn.Linear(state_size + embedding_dimension, state_size)
        self.W_r = nn.Linear(state_size + embedding_dimension, state_size)
        self.W_n = nn.Linear(state_size + embedding_dimension, state_size)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor):
        """Performs a forward pass through the GRU cell.

        Args:
            x (torch.Tensor): The input to the cell.
            hidden_state (torch.Tensor): The hidden state of the previous cell.

        Returns:
            torch.Tensor: The hidden state of the current cell.
        """
        # Calculate the update gate, reset gate and hidden state.
        z = torch.sigmoid(self.W_z(torch.cat((x, hidden_state), dim=1)))
        r = torch.sigmoid(self.W_r(torch.cat((x, hidden_state), dim=1)))
        n = torch.tanh(self.W_n(torch.cat((x, r * hidden_state), dim=1)))

        # Calculate the new hidden state.
        hidden_state = (1 - z) * hidden_state + z * n

        return hidden_state

class GRU(nn.Module):
    def __init__(self, state_size: int, embedding_dimension: int, vocab_size: int, layers: int = 1):
        super().__init__()
        self.state_size = state_size
        self.embedding_dimension = embedding_dimension
        self.layers = layers

        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.cells = nn.ModuleList([GRUCell(state_size, embedding_dimension)] + [GRUCell(state_size, state_size) for _ in range(layers - 1)])
        self.output = nn.Linear(state_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the GRU.

        Args:
            x (torch.Tensor): The input to the GRU.
            hidden_state (torch.Tensor): The hidden state of the previous GRU.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output of the GRU and the hidden state of the current GRU.
        """
        # Initialize the hidden if they are not given.
        if hidden_states is None:
            hidden_states = torch.zeros((x.shape[0], self.state_size), requires_grad=False, device=x.device)
            # Initialize the hidden states for each layer.
            hidden_states = [hidden_states for _ in range(self.layers)]

        # Initialize the outputs.
        outputs = []
        embeddings = self.embedding(x)

        # Iterate over the input.
        for i in range(x.shape[1]):
            # Get the embedding of the current word.
            embedding = embeddings[:, i, :]

            # Iterate over the layers.
            for j in range(self.layers):
                # Calculate the hidden state of this iteration.
                hidden_states[j] = self.cells[j](embedding, hidden_states[j])
                # The embedding of the next layer is the hidden state of this layer.
                embedding = hidden_states[j]

            # Append the output of this iteration.
            outputs.append(self.output(hidden_states[-1]))

        # Stack the outputs.
        outputs = torch.stack(outputs, dim=1)

        # Return the hidden states and the outputs.
        return outputs, hidden_states
