from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time


class RNNCell(nn.Module):
    def __init__(self, state_size: int, input_size: int):
        """Initializes the Recurrent Neural Network cell with the specified arguments.

        Args:
            state_size (int): The size of the hidden state. The hidden state is the output of the previous iteration.
            input_size (int): The size of the input.
        """
        super().__init__()
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


class RNN(nn.Module):
    def __init__(self, state_size: int, embedding_dimension: int, vocabulary_size: int, layers: int = 1):
        super().__init__()
        self.state_size = state_size
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.hidden_states = None

        self.layers = layers
        self.cells = nn.ModuleList([RNNCell(state_size, embedding_dimension) for _ in range(layers)])
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.output = nn.Linear(state_size, vocabulary_size)

    def forward(self, x: torch.Tensor, discard_states = True):
        """Forward pass of the RNN. This function is called when the model is called.
        It takes a batch of sequences and returns the output of the RNN. The output is a tensor of shape (B, T, V)
        where B is the batch size, T is the sequence length and V is the vocabulary size.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T) where B is the batch size and T is the sequence length.
            discard_states (bool, optional): Whether to discard the hidden states after the forward pass. Defaults to True.

        Returns:
            _type_: _description_
        """
        B, T = x.shape

        x = self.embedding(x)

        hidden_states = torch.zeros((B, self.state_size), requires_grad=False, device=x.device) if discard_states or self.hidden_states is None else self.hidden_states
        outputs = torch.zeros((B, T, self.vocabulary_size), requires_grad=False, device=x.device)

        # Walk through the sequence one step at a time, feeding the output of the previous step as input to the next step.
        for t in range(T):
            for layer in range(self.layers):
                hidden_states = self.cells[layer](x[:, t], hidden_states)
                #hidden_states = hidden_states.detach()

            # Transform last hidden layer to the output of the RNN.
            outputs[:, t] = self.output(hidden_states)

        # Save the hidden states for the next forward pass.
        if not discard_states:
            self.hidden_states = hidden_states

        return outputs
    
class RNNTrainer:
    def __init__(self, embedding_dimension: int, context_length: int, layers: int = 1):
        """Initializes an RNN model with the specified arguments.

        Args:
            embedding_dimension (int): The dimension of the embedding.
            context_length (int): The length of the context.
            layers (int, optional): The amount of layers in the model. Defaults to 1.
        """
        nn.RNN
        self.generator = torch.Generator().manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = context_length

        self.model = RNN(500, embedding_dimension, enc.n_vocab + 1, layers) # Additional word for 0 padding.
        self.model.to(self.device)

    def _collate_fn_pad(self, batch):
        """Pad the batch to be of uniform length.
        """
        # Pad tensors to be of uniform length.
        batch = [ torch.Tensor(t) for t in batch ]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

        return batch

    def _prepare(self):
        """Builds the model, fetches the data and splits it.
        """
        # Prepare data.
        recipes = encode_recipes(get_recipes()[:1000])

        # This data is of variable length. It needs to be packed before forward pass.
        data = [torch.tensor(datapoint) for datapoint in create_sliding(recipes, self.context_length + 1, 50259)]

        train_length = int(0.8 * len(data))
        valid_length = int(0.1 * len(data))
        test_length = len(data) - train_length - valid_length

        train_set, valid_set, test_set = torch.utils.data.random_split(data, [train_length, valid_length, test_length], self.generator)
        self.train_set, self.valid_set, self.test_set = [train_set.dataset[i] for i in train_set.indices],\
                                                        [valid_set.dataset[i] for i in valid_set.indices],\
                                                        [test_set.dataset[i] for i in test_set.indices]

    def train(self, max_epochs: int = 20, max_time: int = -1, max_iterations: int = -1, batch_size: int = 8) -> List[List[float]]:
        """Trains the model for as long as the validation loss decreases.

        Args:
            max_epochs (int, optional): The maximum amount of epochs to train for. Defaults to 20.
            max_time (int, optional): The maximum amount of time to train for. Defaults to -1.
            max_iterations (int, optional): The maximum amount of iterations to train for. Defaults to -1.
            batch_size (int, optional): The batch size. Defaults to 8.
        Returns:
            List[List[float]]: The training and validation losses.
        """
        self._prepare()
        self.model.train()

        sampler = torch.utils.data.DataLoader(self.train_set, batch_size, shuffle=True, generator=self.generator, collate_fn=self._collate_fn_pad)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3)
        losses = [[],[]]

        epoch = 1
        iteration = 0
        end_time = time.time() + max_time
        last_state_dict = {}

        while True:
            for batch in tqdm.tqdm(sampler, total=len(sampler), desc=f"Epoch {epoch}"):
                try:
                    self.model.train()
                    
                    # The data was padded to make it loadable. Pack it to ignore padded data.
                    x = batch[:, :-1].to(self.device)
                    y = batch[:, 1:].to(self.device)
                    logits = self.model.forward(x)

                    B, T, C = logits.shape
                    y = y.reshape(B*T)
                    logits = logits.reshape(B*T, C)
                    loss = torch.nn.functional.cross_entropy(logits, y, ignore_index=0) # Only compute loss on non-padded outputs.

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    iteration += 1

                    if iteration == max_iterations or (max_time > 0 and time.time() > end_time):
                        epoch = max_epochs
                        return
                    
                    # Show current performance every once in a while.
                    if iteration % 10 == 0:
                        print(f"Training loss of current epoch: {self.test(test_set=self.train_set[:100], batch_size=batch_size)}")
                        print(f"Validation loss of current epoch: {self.test(test_set=self.valid_set[:100], batch_size=batch_size)}")
                except KeyboardInterrupt as e:
                    print("Saving model...")
                    self.save_model()
                    exit()

            losses[0].append(self.test(test_set=self.train_set[:100], batch_size=batch_size))
            losses[1].append(self.test(test_set=self.valid_set[:100], batch_size=batch_size))
            self.model.train()

            epoch += 1
            gain = (losses[1][-2] - losses[1][-1]) if len(losses[1]) > 1 else 1
            if gain < 0:
                # Validation error increased, load last state and abort.
                print(f"Validation loss increased. Resetting state to last epoch and aborting.")
                self.model.load_state_dict(last_state_dict)
                break
            if epoch > max_epochs:
                break

            last_state_dict = self.model.state_dict()

        return losses
    
    @torch.no_grad()
    def test(self, test_set: Any, batch_size: int = 16) -> float:
        """Tests the current model on the specified dataset.

        Args:
            valid_set (bool, optional): Whether to use the validation set instead of the test set. Defaults to False.
            batch_size (int, optional): The amount of data points to evaluate at once. Defaults to 4096 (fits in roughly 4GB of VRAM).

        Returns:
            float: The cross entropy loss.
        """
        self.model.eval()

        sampler = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True, generator=self.generator, collate_fn=self._collate_fn_pad)
        batches = len(sampler)

        avg_loss = 0
        for batch in sampler:
            # The data was padded to make it loadable. Pack it to ignore padded data.
            x = batch[:, :-1].to(self.device)
            y = batch[:, 1:].to(self.device)
            logits = self.model.forward(x)

            B, T, C = logits.shape
            y = y.reshape(B*T)
            logits = logits.reshape(B*T, C)
            loss = torch.nn.functional.cross_entropy(logits, y, ignore_index=0) # Only compute loss on non-padded outputs.

            avg_loss += loss.item() / batches

        return avg_loss

    @torch.no_grad()
    def generate_recipe(self, ingredients: str = None, print_live: bool = True) -> str:
        """Generates a recipe from a list of ingredients.
        Or generates a complete recipe if none are provided.
        Returns:
            str: The full recipe.
        """
        self.model.eval()

        recipe_codes = []
        recipe_text = "Ingredients:\n"
        # Start with 1 padding.
        context = [50259]

        # Fill context with recipe.
        if ingredients:
            recipe_text = ingredients
            print("\nInstructions:\n", end="")
            ingredients += "<|ingredients_end|>"
            recipe_codes = encode_recipes([ingredients])[0]
            context += recipe_codes
        
        hidden_state = None
        # Fill hidden state with current context.
        for value in context:
            logits, hidden_state = self.model.forward(torch.tensor([[value]]).to(self.device), hidden_state) if hidden_state is not None else\
                                      self.model.forward(torch.tensor([[value]]).to(self.device))

        while True:
            # Model generated result for every index of context. We only need the last one.
            logits, hidden_state = self.model.forward(torch.tensor([[context[-1]]]).to(self.device), hidden_state)
            last_logit = logits[0, 0, :]

            probs = torch.nn.functional.softmax(last_logit, dim=0)
            recipe_code = torch.multinomial(probs, num_samples=1).item()
            
            # Don't add padding.
            if recipe_code == 50259:
                continue
            
            context.append(recipe_code)
            if len(context) > self.context_length:
                context = context[1:]
                
            recipe_codes.append(recipe_code)

            # Make recipe look nicer.
            next_text = enc.decode([recipe_code]).replace("<|padding|>", "")\
                                                 .replace("<|next_step|>", "\n\nNext:\n")\
                                                 .replace("<|ingredients_end|>", "\n\nInstructions:\n")\
                                                 .replace("<|endoftext|>", "\n\n")
            if print_live:
                print(next_text, end="")
            recipe_text += next_text

            # End of text reached.
            if recipe_code == 50256:
                break
        
        return recipe_text

    def save_model(self):
        torch.save(self.model.state_dict(), "./Models/Instructions/RNN.pkl")

    def load_model(self):
        state_dict = torch.load("./Models/Instructions/RNN.pkl")
        self.model.load_state_dict(state_dict)
