from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
import math
from Models.Instructions.ModuleBase import EncoderDecoderModuleBase
import optuna

class EncoderDecoderTransformerTorch(EncoderDecoderModuleBase):
    """The RNN model. This model is a simple RNN with an embedding layer and a linear output layer.
    This is done using the pytorch RNN module, which is much faster than a custom implementation.
    """
    def __init__(self, source_length: int, target_length: int, blocks: int, neurons: int, embedding_dimension: int, heads: int, dropout: float, vocabulary_size: int):
        """Initializes a multi layer perception model with the specified arguments.

        Args:
            blocks (int): The amount of decoder blocks.
            embedding_dimension (int): The dimension of the embedding layer.
            neurons (int): The amount of neurons within the feed forward layer.
            heads (int): The amount of heads.
            dropout (float): The dropout factor for regularization.
            vocabulary_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.save_hyperparameters()
        self.embedding_dimension = embedding_dimension
        self.blocks = blocks
        self.heads = heads
        self.dropout = dropout
        self.neurons = neurons
        self.source_length = source_length
        self.target_length = target_length
        self.embedding_dimension = embedding_dimension
        self.decoder_embedding = nn.Embedding(vocabulary_size, self.embedding_dimension)
        self.encoder_embedding = nn.Embedding(vocabulary_size, self.embedding_dimension)
        self.transformer = nn.Transformer(d_model=embedding_dimension, nhead=heads, num_encoder_layers=blocks, num_decoder_layers=blocks, dim_feedforward=neurons, dropout=dropout, activation='relu', batch_first=True)
        self.mask = nn.Parameter(self.transformer.generate_square_subsequent_mask(target_length), requires_grad=False)
        # Positional encoding for the source input.
        self.pos_encodings = nn.Parameter(self._get_pos_encoding(max(source_length, target_length)), requires_grad=False)
        self.norm = nn.LayerNorm(self.embedding_dimension)
        self.linear = nn.Linear(self.embedding_dimension, vocabulary_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            hidden_states (torch.Tensor, optional): The hidden states. Defaults to None.
            cell_states (torch.Tensor, optional): The cell states. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The output, hidden states and cell states.
        """
        #src_padding_mask = src == 0
        #src_padding_mask[:, 0] = False
        src = self.encoder_embedding(src)

        #tgt_padding_mask = tgt == 0
        #tgt_padding_mask[:, 0] = False
        tgt = self.decoder_embedding(tgt)

        B, T, C = src.shape
        src = src + self.pos_encodings[:T]
        B, T, C = tgt.shape
        tgt = tgt + self.pos_encodings[:T]
        x = self.transformer(src, tgt, tgt_mask=self.mask[:T, :T])#, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        x = self.norm(x)
        x = self.linear(x)
        return x
    
    def _get_pos_encoding(self, seq_length):
        """Returns the positional encoding for the specified sequence length.

        Args:
            seq_length (int): The length of the sequence.
            device (torch.device): The device on which the positional encoding should be stored.

        Returns:
            torch.Tensor: The positional encoding.
        """
        pos = torch.arange(seq_length, dtype=torch.float).reshape(-1, 1)
        div_term = torch.exp(torch.arange(0, self.embedding_dimension, 2, dtype=torch.float) * -(math.log(10000.0) / self.embedding_dimension))
        pos_encodings = torch.zeros(seq_length, self.embedding_dimension)
        pos_encodings[:, 0::2] = torch.sin(pos * div_term)
        pos_encodings[:, 1::2] = torch.cos(pos * div_term)
        return pos_encodings
    
    def get_optuna_parameters(self, trial: optuna.Trial) -> List[Any]:
        """Gets the parameters to optimize using optuna.

        Args:
            trial (optuna.Trial): The trial.

        Returns:
            List[Any]: The parameters for the next objective step.
        """
        blocks: int = trial.suggest_int("blocks", 1, 10)
        neurons: int = trial.suggest_int("neurons", 1, 1000)
        heads: int = trial.suggest_int("heads", 1, 10)
        # Embedding dimension must be divisible by the amount of heads, so set step to heads.
        embedding_dimension: int = trial.suggest_int("embedding_dimension", heads, 1000, step=heads)
        dropout: float = trial.suggest_float("dropout", 0.0, 0.4)

        return [self.hparams.source_length, self.hparams.target_length, blocks, neurons, embedding_dimension, heads, dropout, self.hparams.vocabulary_size]