from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
from Models.Instructions.ModuleBase import ModuleBase


class RNNCell(nn.Module):
    def __init__(self, state_size: int, input_size: int):
        """Initializes the Recurrent Neural Network cell with the specified arguments.

        Args:
            state_size (int): The size of the hidden state. The hidden state is the output of the previous iteration.
            input_size (int): The size of the input.
        """
        super().__init__()
        self.save_hyperparameters()
        self.state_size = state_size
        self.input_size = input_size

        self.input_transform = nn.Linear(input_size, state_size, bias=False)
        self.hidden_transform = nn.Linear(state_size, state_size, bias=False)
        self.norm = nn.LayerNorm(state_size)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor):
        x = self.input_transform(x)
        hidden = self.hidden_transform(hidden_state)
        hidden = torch.tanh(self.norm(x + hidden))

        return hidden


class RNN(ModuleBase):
    def __init__(self, state_size: int, embedding_dimension: int, vocabulary_size: int, layers: int = 1):
        super().__init__()
        self.state_size = state_size
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.hidden_states = None

        self.layers = layers
        self.cells = nn.ModuleList([RNNCell(state_size, embedding_dimension)] + [RNNCell(state_size, state_size) for _ in range(layers - 1)])
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.output = nn.Linear(state_size, vocabulary_size)

    def forward(self, x: torch.Tensor, hidden_states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass of the RNN. This function is called when the model is called.
        It takes a batch of sequences and returns the output of the RNN. The output is a tensor of shape (B, T, V)
        where B is the batch size, T is the sequence length and V is the vocabulary size.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T) where B is the batch size and T is the sequence length.
            discard_states (bool, optional): Whether to discard the hidden states after the forward pass. Defaults to True.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The output of the RNN and the hidden states for each layer.
        """
        # Initialize the hidden states if they are not given.
        if hidden_states is None:
            hidden_states = torch.zeros((x.shape[0], self.state_size), requires_grad=False, device=x.device)

            # Initialize the hidden states for each layer.
            hidden_states = [hidden_states for _ in range(self.layers)]

        # Initialize the outputs.
        outputs = []

        # Iterate over the input.
        for i in range(x.shape[1]):
            # Get the embedding of the current word.
            embedding = self.embedding(x[:, i])

            # Iterate over the layers.
            for j in range(self.layers):
                # Calculate the hidden state and the cell state of this iteration.
                hidden_states[j] = self.cells[j](embedding, hidden_states[j])
                # The embedding of the next layer is the hidden state of this layer.
                embedding = hidden_states[j]

            # Append the output of this iteration.
            outputs.append(self.output(hidden_states[-1]))
        
        # Stack the outputs.
        outputs = torch.stack(outputs, dim=1)

        # Return the hidden states, the cell states and the outputs.
        return outputs, hidden_states
