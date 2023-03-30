from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, embedding_dimension, dropout: float):
        super().__init__()
        self.head_size = head_size
        self.embedding_dimension = embedding_dimension

        # Every token can learn to look (query) for any kind of token (key), which will retrieve some value.
        self.key = nn.Linear(self.embedding_dimension, self.head_size, bias=False)
        self.query = nn.Linear(self.embedding_dimension, self.head_size, bias=False)
        self.value = nn.Linear(self.embedding_dimension, self.head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(embedding_dimension, embedding_dimension)))

    def forward(self, x):
        B, T, C = x.shape

        key = self.key.forward(x)
        query = self.query.forward(x)
        
        # Attention weights.
        weights = query @ key.transpose(-1, -2) * C **-0.5 # Prevent weights from becoming one-hot.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # Decoder. Ensure future tokens aren't taken into account.
        weights = nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        values = self.value(x)
        return weights @ values
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, head_size, embedding_dimension, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size, embedding_dimension, dropout) for i in range(heads)])
        self.projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the results of all heads.
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, context_length: int, neurons: int, embedding_dimension: int, dropout: float):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dimension, neurons),
            nn.ReLU(),
            nn.Linear(neurons, embedding_dimension),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.model(x)

class DecoderBlock(nn.Module):
    def __init__(self, context_length: int, embedding_dimension: int, neurons: int, heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadedAttention(heads, embedding_dimension // heads, embedding_dimension, dropout)
        self.feed_forward = FeedForward(context_length, neurons, embedding_dimension, dropout)
        self.attention_layernorm = nn.LayerNorm(embedding_dimension)
        self.feed_forward_layernorm = nn.LayerNorm(embedding_dimension)

    def forward(self, x):
        x = x + self.attention(self.attention_layernorm(x))
        x = x + self.feed_forward(self.feed_forward_layernorm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, context_length: int, blocks: int, neurons: int, embedding_dimension: int, heads: int, dropout: float, vocabulary_size: int):
        """Initializes a multi layer perception model with the specified arguments.

        Args:
            context_length (int): The context length the predict the next ingredient with.
            blocks (int): The amount of blocks within the transformer.
            neurons (int): The amount of neurons within the feed forward layer.
            embedding_dimension (int): The dimension of the embedding layer.
            heads (int): The amount of heads in the multi head attention layer.
            dropout (float): The dropout factor for regularization.
            vocabulary_size (int): The amount of words to learn.
        """
        super().__init__()
        self.context_length = context_length
        self.embedding_dimension = embedding_dimension
        self.neurons = neurons
        self.blocks = blocks
        self.heads = heads
        self.dropout = dropout
        self.positional_embedding = nn.Embedding(self.context_length, self.embedding_dimension)
        self.model = nn.Sequential()
        self.embedding = nn.Embedding(vocabulary_size, self.embedding_dimension)
        self.register_buffer("positions", torch.arange(context_length))

        # Create blocks.
        for i in range(self.blocks):
            self.model.add_module(f"decoderblock_{i}", DecoderBlock(self.context_length, self.embedding_dimension, self.neurons, self.heads, self.dropout))
        
        self.model.add_module(f"layer_norm_out", nn.LayerNorm(self.embedding_dimension))
        self.model.add_module(f"linear_out", nn.Linear(self.embedding_dimension, vocabulary_size))
    
    def forward(self, x):
        B, T = x.shape
        positional_embedding = self.embedding(x)
        positional_embedding += self.positional_embedding(self.positions)
        out = self.model.forward(positional_embedding)

        return out


class TransformerTrainer:
    def __init__(self, context_length: int, blocks: int, neurons: int, embedding_dimension: int, heads: int, dropout: float):
        """Initializes a transformer model with the specified arguments.

        Args:
            context_length (int): The context length the predict the next ingredient with.
            blocks (int): The amount of blocks within the transformer.
            neurons (int): The amount of neurons within the feed forward layer.
            embedding_dimension (int): The dimension of the embedding layer.
            heads (int): The amount of heads in the multi head attention layer.
            dropout (float): The dropout factor for regularization.
        """
        self.generator = torch.Generator().manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = context_length

        self.model = Transformer(context_length, blocks, neurons, embedding_dimension, heads, dropout, enc.n_vocab)
        self.model.to(self.device)

    def _prepare(self):
        """Builds the model, fetches the data and splits it.
        """
        # Prepare data.
        recipes = encode_recipes(get_recipes()[:5000])
        data = torch.tensor(create_ngram(recipes, self.context_length + 1, 50259, False))
        train_length = int(0.8 * len(data))
        valid_length = int(0.1 * len(data))
        test_length = len(data) - train_length - valid_length

        train_set, valid_set, test_set = torch.utils.data.random_split(data, [train_length, valid_length, test_length], self.generator)
        self.train_set, self.valid_set, self.test_set = train_set.dataset[train_set.indices], valid_set.dataset[valid_set.indices], test_set.dataset[test_set.indices]

    def train(self, max_epochs: int = 20, max_time: int = -1, max_iterations: int = -1, batch_size: int = 16) -> List[List[float]]:
        """Trains the model for as long as the validation loss decreases.

        Args:
            max_epochs (int, optional): The maximum amount of epochs to train for. The model will abort if the validation loss gain gets too small.
            max_time (int, optional): The maximum amount of seconds to train for. If the training takes longer, it breaks.
            max_iterations (int, optional): The maximum amount batches to train on.
            batch_size (int, optional): The amount of data points to evaluate at once. Defaults to 32.

        Returns:
            List[List[float]]: The training and validation losses.
        """
        self._prepare()
        self.model.train()

        sampler = torch.utils.data.DataLoader(self.train_set, batch_size, shuffle=True, generator=self.generator)
        batches = len(sampler)
    
        optimizer = torch.optim.AdamW(self.model.parameters(), eps=1e-8)
        losses = [[],[]]

        epoch = 1
        iteration = 0
        end_time = time.time() + max_time
        last_state_dict = {}

        while True:
            for batch in tqdm.tqdm(sampler, total=len(sampler), desc=f"Epoch {epoch}"):
                optimizer.zero_grad()

                x = batch[:, :-1].to(self.device)
                y = batch[:, 1:].to(self.device)

                logits = self.model.forward(x)
                B, T, C = logits.shape
                targets = y.view(B*T)
                logits = logits.view(B*T, C)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                losses[0].append(loss.item())
                loss.backward()

                optimizer.step()

                iteration += 1
                if iteration == max_iterations or (max_time > 0 and time.time() > end_time):
                    print(f"Average training loss: {sum(losses[0][-100:]) / (100)}")
                    return

            losses[0].append(self.test(test_set=self.train_set, batch_size=batch_size))
            losses[1].append(self.test(test_set=self.valid_set, batch_size=batch_size))
            print(f"Training loss of current epoch: {losses[0][-1]}")
            print(f"Validation loss of current epoch: {losses[1][-1]}")

            epoch += 1
            gain = (losses[1][-2] - losses[1][-1]) if len(losses[1]) > 1 else 1
            if gain < 0:
                # Validation error increased, load last state and abort.
                print(f"Validation loss increased. Resetting state to last epoch and aborting.")
                self.model.load_state_dict(last_state_dict)
                break
            if epoch >= max_epochs:
                break

            last_state_dict = self.model.state_dict()

        return losses
    
    @torch.no_grad()
    def test(self, test_set: Any, batch_size: int = 32) -> float:
        """Tests the current model on the specified dataset.

        Args:
            valid_set (bool, optional): Whether to use the validation set instead of the test set. Defaults to False.
            batch_size (int, optional): The amount of data points to evaluate at once. Defaults to 4096 (fits in roughly 4GB of VRAM).

        Returns:
            float: The cross entropy loss.
        """
        self.model.eval()

        sampler = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True, generator=self.generator)
        batches = len(sampler)

        avg_loss = 0
        for batch in sampler:
            x = batch[:, :-1].to(self.device)
            y = batch[:, 1:].to(self.device)

            logits = self.model.forward(x)
            B, T, C = logits.shape
            targets = y.view(B*T)
            logits = logits.view(B*T, C)
            
            loss = torch.nn.functional.cross_entropy(logits, targets)
            avg_loss += loss.item() / batches

        return avg_loss

    @torch.no_grad()
    def generate_recipe(self, ingredients: str = None) -> str:
        """Generates a recipe from a list of ingredients.
        Or generates a complete recipe if none are provided.

        Here is a sample after training for 5 minutes (~650 batches, 200 context, 3 blocks, 500 neurons, 400 embed_dim, 4 heads, 0 dropout):

            Ingredients:
            sugar, eggs, flour
            Instructions:
            preheat oven to 375 degrees
            Next:
            400 degrees f in a large mixing bowl
            Next:
            fold in the flour
            Next:
            press to use a taste in butter or cool sugar
            Next:
            add vanilla and milk , then mix until light lightly brown sugar
            Next:
            in to preventnuts , water
            Next:
            press fat from the flour , about 1 / 4-inch square and keep in 3 minutes , or until smooth , then about salt in each through , 1 / 2 strips amish with 1 cup of 155
            Next:
            remove from heat to taste through
            Next:
            whisk in the batter , makingfing pepper
            Next:
            in dip in ice cream , mixing bowl , beat eggs
            Next:
            stir in apple , sugar , stirring frequently , oil of requirement

        Returns:
            str: The full recipe.
        """
        self.model.eval()

        recipe = []
        context = [50259] * self.context_length

        # Fill context will recipe.
        if ingredients:
            ingredients += "<|ingredients_end|>"
            recipe = encode_recipes([ingredients])[0]
            for i, word in enumerate(recipe):
                word_index = self.context_length - len(recipe) + i
                if word_index >= 0:
                    context[word_index] = word

        while True:
            # Model generated result for every index of context. We only need the last one.
            logits = self.model.forward(torch.tensor([context]).to(self.device))
            last_logit = logits[:, -1, :]

            probs = torch.nn.functional.softmax(last_logit, dim=1)
            recipe_code = torch.multinomial(probs, num_samples=1).item()
            
            # Don't add padding.
            if recipe_code == 50259:
                continue
            
            context = context[1:] + [recipe_code]
            recipe.append(recipe_code)

            # End of text reached.
            if recipe_code == 50256:
                break
        
        # Make recipe look nicer.
        return "Ingredients:\n" + \
               decode_recipes([recipe])[0].replace("<|endoftext|>", "")\
                                          .replace("<|padding|>", "")\
                                          .replace("<|next_step|>", "\n\nNext:\n")\
                                          .replace("<|ingredients_end|>", "\n\nInstructions:\n")

    def save_model(self):
        torch.save(self.model.state_dict(), "./Models/Instructions/Transformer.pkl")

    def load_model(self):
        state_dict = torch.load("./Models/Instructions/Transformer.pkl")
        self.model.load_state_dict(state_dict)
