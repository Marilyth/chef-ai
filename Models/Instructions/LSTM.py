from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
from Models.Instructions.ModuleBase import ModuleBase


class LSTMCell(nn.Module):
    def __init__(self, state_size: int, input_size: int):
        """Initializes the LSTM cell with the specified arguments.

        Args:
            state_size (int): The size of the hidden state. The hidden state is the output of the previous iteration.
            input_size (int): The size of the input.
        """
        super().__init__()
        self.state_size = state_size
        self.input_size = input_size

        # This matrix is the concatenation of the input gate, the forget gate, the output gate and the cell state gate below.
        # This improves the performance of the LSTM cell.
        self.matrices = nn.Linear(input_size, state_size * 4, bias=True)
        self.h_matrices = nn.Linear(state_size, state_size * 4, bias=False)

        # Input gate. This gate determines how much of the input is used to update the cell state.
        #self.i = nn.Linear(input_size, state_size, bias=False)
        #self.h_i = nn.Linear(state_size, state_size, bias=False)

        # Forget gate. This gate determines how much of the previous cell state is kept.
        #self.f = nn.Linear(input_size, state_size, bias=False)
        #self.h_f = nn.Linear(state_size, state_size, bias=False)

        # Output gate. This gate determines how much of the hidden state is used as output.
        #self.o = nn.Linear(input_size, state_size, bias=False)
        #self.h_o = nn.Linear(state_size, state_size, bias=False)

        # This is used in addition with the input gate to update the cell state.
        #self.g = nn.Linear(input_size, state_size, bias=False)
        #self.h_g = nn.Linear(state_size, state_size, bias=False)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None, cell_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the LSTM cell. This function is called by the LSTM class and calculates the hidden state and the cell state of this iteration.
        The hidden state is the output of this iteration and the cell state is the state of the cell.
        Intuitively, the cell state is the long term memory of the cell and the hidden state is the short term memory of the cell.

        Args:
            x (torch.Tensor): The input to the cell.
            hidden_state (torch.Tensor): The hidden state of the previous iteration. Defaults to None.
            cell_state (torch.Tensor, optional): The cell state of the previous iteration. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The hidden state and the cell state of this iteration.
        """
        # Initialize the hidden state and the cell state if they are not given.
        if hidden_state is None:
            hidden_state = torch.zeros((x.shape[0], self.state_size), requires_grad=False, device=x.device)
        if cell_state is None:
            cell_state = torch.zeros((x.shape[0], self.state_size), requires_grad=False, device=x.device)

        gates = self.matrices(x) + self.h_matrices(hidden_state)

        # Calculate the input gate.
        i = torch.sigmoid(gates[:, :self.state_size])
        #i = torch.sigmoid(self.i(x) + self.h_i(hidden_state))
        # Calculate the forget gate.
        f = torch.sigmoid(gates[:, self.state_size:self.state_size * 2])
        #f = torch.sigmoid(self.f(x) + self.h_f(hidden_state))
        # Calculate the output gate.
        o = torch.sigmoid(gates[:, self.state_size * 2:self.state_size * 3])
        #o = torch.sigmoid(self.o(x) + self.h_o(hidden_state))
        # Calculate the cell state.
        g = torch.tanh(gates[:, self.state_size * 3:])
        #g = torch.tanh(self.g(x) + self.h_g(hidden_state))

        # Calculate the new cell state.
        cell_state = f * cell_state + i * g

        # Calculate the new hidden state.
        hidden_state = o * torch.tanh(cell_state)

        return hidden_state, cell_state 


class LSTM(ModuleBase):
    def __init__(self, state_size: int, embedding_dimension: int, vocabulary_size: int, layers: int = 1):
        """Initializes the LSTM with the specified arguments.
        The LSTM is a recurrent neural network that uses a cell state to remember information over time.

        Args:
            state_size (int): The size of the hidden state. The hidden state is the output of the previous iteration.
            embedding_dimension (int): The dimension of the embedding.
            vocabulary_size (int): The size of the vocabulary.
            layers (int, optional): The number of layers. Defaults to 1.
        """
        super().__init__()
        self.save_hyperparameters()
        self.state_size = state_size
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size

        self.layers = layers

        # Create the LSTM cells. The first cell takes the input and the embedding as input. The other cells take the output of the previous cell as input.
        self.cells = nn.ModuleList([LSTMCell(state_size, embedding_dimension)] + [LSTMCell(state_size, state_size) for _ in range(layers - 1)])

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.output = nn.Linear(state_size, vocabulary_size)

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor = None, cell_states: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The forward pass of the LSTM. This function is calls the LSTM cells and calculates the hidden state and the cell state for each iteration.

        Args:
            x (torch.Tensor): The input to the LSTM.
            hidden_states (torch.Tensor, optional): The hidden states of the previous iteration. Defaults to None.
            cell_states (torch.Tensor, optional): The cell states of the previous iteration. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The hidden states, the cell states and the outputs of the LSTM.
        """
        # Initialize the hidden states and the cell states if they are not given.
        if hidden_states is None or cell_states is None:
            hidden_states = torch.zeros((x.shape[0], self.state_size), requires_grad=False, device=x.device)
            cell_states = torch.zeros((x.shape[0], self.state_size), requires_grad=False, device=x.device)

            # Initialize the hidden states and the cell states for each layer.
            hidden_states = [hidden_states for _ in range(self.layers)]
            cell_states = [cell_states for _ in range(self.layers)]

        # Initialize the outputs.
        outputs = []

        # Iterate over the input.
        for i in range(x.shape[1]):
            # Get the embedding of the current word.
            embedding = self.embedding(x[:, i])

            # Iterate over the layers.
            for j in range(self.layers):
                # Calculate the hidden state and the cell state of this iteration.
                hidden_states[j], cell_states[j] = self.cells[j](embedding, hidden_states[j], cell_states[j])
                # The embedding of the next layer is the hidden state of this layer.
                embedding = hidden_states[j]

            # Append the output of this iteration.
            outputs.append(self.output(hidden_states[-1]))

        # Stack the outputs.
        outputs = torch.stack(outputs, dim=1)

        # Return the hidden states, the cell states and the outputs.
        return outputs, hidden_states, cell_states
