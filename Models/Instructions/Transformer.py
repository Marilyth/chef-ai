from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
import optuna
from Models.Instructions.ModuleBase import DecoderOnlyBase


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, embedding_dimension, dropout: float):
        super().__init__()
        self.head_size = head_size
        self.embedding_dimension = embedding_dimension

        # Every token can learn to look (query) for any kind of token (key), which will retrieve some value.
        self.key = nn.Linear(self.embedding_dimension, self.head_size, bias=False)
        self.query = nn.Linear(self.embedding_dimension, self.head_size, bias=False)
        self.value = nn.Linear(self.embedding_dimension, self.head_size, bias=False)
        self.dropout = nn.Dropout(dropout) # Dropout to prevent overfitting.

        # Mask to prevent the model from looking at future tokens.
        # This is done by creating a triangular matrix with ones on the lower diagonal.
        # This matrix is then multiplied with the attention weights.
        # This will ensure that the attention weights of future tokens are set to zero.
        self.register_buffer("tril", torch.tril(torch.ones(embedding_dimension, embedding_dimension)))

    def forward(self, x: torch.Tensor):
        """Forward pass of the self attention head.
        
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        B, T, C = x.shape # Batch size, sequence length, embedding dimension.

        key = self.key.forward(x)
        query = self.query.forward(x)
        
        # Attention weights.
        weights = query @ key.transpose(-1, -2) * C **-0.5 # Prevent weights from becoming one-hot.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # Mask future tokens.
        weights = nn.functional.softmax(weights, dim=-1) # Softmax to ensure the weights sum up to one.
        weights = self.dropout(weights) # Dropout to prevent overfitting.

        values = self.value(x)
        return weights @ values
    
class MultiHeadedAttention(nn.Module):
    """Multi headed attention layer. This layer consists of multiple self attention heads.
    The results of these heads are concatenated and projected to the embedding dimension. This is done to increase the model's capacity.

    Args:
        heads (int): The amount of heads.
        head_size (int): The size of each head.
        embedding_dimension (int): The dimension of the embedding layer.
        dropout (float): The dropout factor for regularization.
    """
    def __init__(self, heads, head_size, embedding_dimension, dropout: float):
        super().__init__()
        # Ensure that the embedding dimension is divisible by the amount of heads.
        assert embedding_dimension % heads == 0
        self.n_heads = heads
        self.dropout = dropout

        #self.heads = nn.ModuleList([SelfAttentionHead(head_size, embedding_dimension, dropout) for i in range(heads)])

        # Key, query and value projections in one linear layer.
        self.causal_attention = nn.Linear(embedding_dimension, 3 * embedding_dimension)
        self.projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multi headed attention layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Concatenate the results of all heads.
        B, T, C = x.shape # Batch size, sequence length, embedding dimension.
        q, k, v = self.causal_attention(x).chunk(3, dim=-1) # Split the concatenated tensor into three tensors.

        # Split the tensors into multiple heads.
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        # Concatenate the heads.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout_layer(self.projection(out))
        return out

class FeedForward(nn.Module):
    """Feed forward layer. This layer consists of two linear layers with a ReLU activation function in between.
    ReLU is used to introduce non-linearity to the model. Dropout is used to prevent overfitting and improve generalization.
    """
    def __init__(self, embedding_dimension: int, neurons: int, dropout: float):
        """Initializes a feed forward layer with the specified arguments.

        Args:
            embedding_dimension (int): The dimension of the embedding layer.
            neurons (int): The amount of neurons within the feed forward layer.
            dropout (float): The dropout factor for regularization.
        """
        super().__init__()
        # Two linear layers with a ReLU activation function in between. Dropout is used to prevent overfitting.
        self.model = nn.Sequential(
            nn.Linear(embedding_dimension, neurons),
            nn.ReLU(),
            nn.Linear(neurons, embedding_dimension),
            nn.Dropout(dropout)
            )

    def forward(self, x: torch.Tensor):
        """Forward pass of the feed forward layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model(x)
    

class DecoderBlock(nn.Module):
    """Decoder block. This block consists of a multi headed attention layer and a feed forward layer.
    Layernorm is used to normalize the output of the attention layer and the feed forward layer because it improves the training process.
    """
    
    def __init__(self, neurons: int, embedding_dimension: int, heads: int, dropout: float):
        """Initializes a decoder block with the specified arguments.

        Args:
            embedding_dimension (int): The dimension of the embedding layer.
            neurons (int): The amount of neurons within the feed forward layer.
            heads (int): The amount of heads.
            dropout (float): The dropout factor for regularization.
        """
        super().__init__()
        
        # Multi headed attention layer. The embedding dimension is divided by the amount of heads to ensure that the concatenated heads have the same dimension as the embedding dimension.
        self.attention = MultiHeadedAttention(heads, embedding_dimension // heads, embedding_dimension, dropout)
        self.feed_forward = FeedForward(embedding_dimension, neurons, dropout)
        self.attention_layernorm = nn.LayerNorm(embedding_dimension)
        self.feed_forward_layernorm = nn.LayerNorm(embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder block. The output of the attention layer and the feed forward layer are added to the input.
        This is done to ensure that the input is not lost. The output of the attention layer and the feed forward layer are normalized with layernorm.
        This is done to improve the training process.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x + self.attention(self.attention_layernorm(x))
        x = x + self.feed_forward(self.feed_forward_layernorm(x))
        return x

class Transformer(DecoderOnlyBase):
    """Transformer model. This model consists of an embedding layer, a positional encoding layer, multiple decoder blocks and a linear layer. 
    The embedding layer is used to learn the representation of the words. The positional encoding layer is used to learn the position of the words.
    The decoder blocks are used to learn the relationships between the words. The linear layer is used to predict the next word.
    """
    def __init__(self, context_length: int, blocks: int, neurons: int, embedding_dimension: int, heads: int, dropout: float, vocabulary_size: int):
        """Initializes a multi layer perception model with the specified arguments.

        Args:
            context_length (int): The length of the context. This is the amount of words that are used to predict the next word.
            blocks (int): The amount of decoder blocks.
            embedding_dimension (int): The dimension of the embedding layer.
            neurons (int): The amount of neurons within the feed forward layer.
            heads (int): The amount of heads.
            dropout (float): The dropout factor for regularization.
            vocabulary_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.save_hyperparameters()
        self.context_length = context_length
        self.embedding_dimension = embedding_dimension
        self.blocks = blocks
        self.heads = heads
        self.dropout = dropout
        self.neurons = neurons
        self.positional_encoding = nn.Embedding(self.context_length, self.embedding_dimension)
        self.model = nn.Sequential()
        self.embedding = nn.Embedding(vocabulary_size, self.embedding_dimension)
        self.register_buffer("positions", torch.arange(context_length))

        # Add the decoder blocks to the model.
        for i in range(self.blocks):
            self.model.add_module(f"decoderblock_{i}", DecoderBlock(self.neurons, self.embedding_dimension, self.heads, self.dropout))
        
        # Add normalization and linear layer to the model. The linear layer is used to predict the next word.
        self.model.add_module(f"layer_norm_out", nn.LayerNorm(self.embedding_dimension))
        self.model.add_module(f"linear_out", nn.Linear(self.embedding_dimension, vocabulary_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer model. The input is first embedded and then the positional encoding is added to the input.
        Afterwards the decoder blocks are applied to the input. The output of the decoder blocks is normalized and then passed through a linear layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, context_length).

        Returns:
            torch.Tensor: The output tensor.
        """
        B, T = x.shape
        positional_embedding = self.embedding(x)
        # The positional encoding is added to the input to learn the position of the words.
        positional_embedding += self.positional_encoding(torch.arange(T, device=x.device))

        return self.model.forward(positional_embedding)

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
        dropout: float = trial.suggest_float("dropout", 0.0, 0.5)

        return [self.hparams.context_length, blocks, neurons, embedding_dimension, heads, dropout, self.hparams.vocabulary_size]
