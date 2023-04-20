from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
import math


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
    def __init__(self, heads, head_size, embedding_dimension, dropout: float, is_causal: bool = False):
        super().__init__()
        # Ensure that the embedding dimension is divisible by the amount of heads.
        assert embedding_dimension % heads == 0
        self.n_heads = heads
        self.dropout = dropout
        self.is_causal = is_causal

        #self.heads = nn.ModuleList([SelfAttentionHead(head_size, embedding_dimension, dropout) for i in range(heads)])

        # Key, query and value projections in one linear layer.
        self.self_attention = nn.Linear(embedding_dimension, 3 * embedding_dimension)
        self.projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the multi headed attention layer.

        Args:
            x (torch.Tensor): The input tensor.
            encoder_output (torch.Tensor): The output of the encoder.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Concatenate the results of all heads.
        if encoder_output is not None:
            x = torch.cat((encoder_output, x), dim=1)

        B, T, C = x.shape # Batch size, sequence length, embedding dimension.
        q, k, v = self.self_attention(x).chunk(3, dim=-1) # Split the concatenated tensor into three tensors.

        # Split the tensors into multiple heads.
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal)
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

class EncoderBlock(nn.Module):
    """Encoder block. This block consists of a multi headed attention layer, a feed forward layer and a residual connection.
    Layernorm is used to normalize the output of the attention layer and the feed forward layer because it improves the training process.
    """
    def __init__(self, neurons: int, embedding_dimension: int, heads: int, dropout: float):
        """Initializes an encoder block with the specified arguments.

        Args:
            embedding_dimension (int): The dimension of the embedding layer.
            neurons (int): The amount of neurons within the feed forward layer.
            heads (int): The amount of heads.
            dropout (float): The dropout factor for regularization.
        """
        super().__init__()
        # Multi headed attention layer. The embedding dimension is divided by the amount of heads to ensure that the concatenated heads have the same dimension as the embedding dimension.
        self.attention = MultiHeadedAttention(heads, embedding_dimension // heads, embedding_dimension, dropout, is_causal=False)
        self.feed_forward = FeedForward(embedding_dimension, neurons, dropout)
        self.layernorm1 = nn.LayerNorm(embedding_dimension)
        self.layernorm2 = nn.LayerNorm(embedding_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """Forward pass of the encoder block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Attention layer.
        attention = self.attention(x)
        # Residual connection.
        attention = self.dropout1(attention) + x
        # Normalize the output of the attention layer.
        attention = self.layernorm1(attention)

        # Feed forward layer.
        feed_forward = self.feed_forward(attention)
        # Residual connection.
        feed_forward = self.dropout2(feed_forward) + attention
        # Normalize the output of the feed forward layer.
        feed_forward = self.layernorm2(feed_forward)

        return feed_forward

class DecoderBlock(nn.Module):
    """Decoder block. This block consists of a masked multi headed attention layer, a multi headed attention layer, a feed forward layer and a residual connection.
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
        self.attention1 = MultiHeadedAttention(heads, embedding_dimension // heads, embedding_dimension, dropout, is_causal=True)
        self.attention2 = MultiHeadedAttention(heads, embedding_dimension // heads, embedding_dimension, dropout, is_causal=False)
        self.feed_forward = FeedForward(embedding_dimension, neurons, dropout)
        self.layernorm1 = nn.LayerNorm(embedding_dimension)
        self.layernorm2 = nn.LayerNorm(embedding_dimension)
        self.layernorm3 = nn.LayerNorm(embedding_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        """Forward pass of the decoder block.

        Args:
            x (torch.Tensor): The input tensor.
            encoder_output (torch.Tensor): The output of the encoder.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Masked attention layer.
        attention1 = self.attention1(x)
        # Residual connection.
        attention1 = self.dropout1(attention1) + x
        # Normalize the output of the attention layer.
        attention1 = self.layernorm1(attention1)

        # Attention layer. The encoder output is used as the key and value. This is known as cross attention.
        attention2 = self.attention2(attention1, encoder_output)
        # Residual connection.
        attention2 = self.dropout2(attention2) + attention1
        # Normalize the output of the attention layer.
        attention2 = self.layernorm2(attention2)

        # Feed forward layer.
        feed_forward = self.feed_forward(attention2)
        # Residual connection.
        feed_forward = self.dropout3(feed_forward) + attention2
        # Normalize the output of the feed forward layer.
        feed_forward = self.layernorm3(feed_forward)

        return feed_forward

class EncoderDecoderTransformer(nn.Module):
    """Encoder decoder transformer model. This model consists of an embedding layer, an encoder and a decoder.
    """
    def __init__(self, blocks: int, neurons: int, embedding_dimension: int, heads: int, dropout: float, vocabulary_size: int):
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
        self.embedding_dimension = embedding_dimension
        self.blocks = blocks
        self.heads = heads
        self.dropout = dropout
        self.neurons = neurons
        self.embedding = nn.Embedding(vocabulary_size, self.embedding_dimension)
        self.encodings = nn.ModuleList([EncoderBlock(self.neurons, self.embedding_dimension, self.heads, self.dropout) for _ in range(blocks)])
        self.decodings = nn.ModuleList([DecoderBlock(self.neurons, self.embedding_dimension, self.heads, self.dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(self.embedding_dimension)
        self.linear = nn.Linear(self.embedding_dimension, vocabulary_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer model. The input is first embedded and then the positional encoding is added to the input.
        Afterwards the decoder blocks are applied to the input. The output of the decoder blocks is normalized and then passed through a linear layer.

        Args:
            src (torch.Tensor): The input tensor.
            tgt (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        B, T = src.shape
        embedding = self.embedding(src)

        # Add positional encoding to the input using the sine and cosine functions.
        seq_length = src.size(1)
        pos_encodings = self._get_pos_encoding(seq_length, embedding.device)
        input_embeddings = embedding + pos_encodings

        # Add positional encoding to the tgt.
        seq_length = tgt.size(1)
        pos_encodings = self._get_pos_encoding(seq_length, embedding.device)
        target_embedding = self.embedding(tgt) + pos_encodings

        # Apply the encoder blocks.
        for i in range(self.blocks):
            input_embeddings = self.encodings[i](input_embeddings)
        
        # Apply the decoder blocks.
        for i in range(self.blocks):
            target_embedding = self.decodings[i](target_embedding, input_embeddings)
        
        # Normalize the output of the decoder blocks.
        target_embedding = self.norm(target_embedding)

        # Pass the output of the decoder blocks through a linear layer.
        output = self.linear(target_embedding)

        return output
    
    def _get_pos_encoding(self, seq_length, device):
        """Returns the positional encoding for the specified sequence length.

        Args:
            seq_length (int): The length of the sequence.
            device (torch.device): The device on which the positional encoding should be stored.

        Returns:
            torch.Tensor: The positional encoding.
        """
        pos_encodings = torch.zeros(seq_length, self.embedding_dimension).to(device)
        for pos in range(seq_length):
            for i in range(0, self.embedding_dimension, 2):
                pos_encodings[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embedding_dimension)))
                pos_encodings[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embedding_dimension)))
        return pos_encodings
