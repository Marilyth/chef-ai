from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm


class MLP:
    def __init__(self, context_length: int, hidden_layers: int, neurons: int, embedding_dimension: int):
        """Initializes a multi layer perception model with the specified arguments.

        Args:
            context_length (int): The context length the predict the next ingredient with.
            hidden_layers (int): The amount of hidden layers in the MLP.
            neurons (int): The amount of neurons within the MLP's hidden layers.
            embedding_dimension (int): The dimension of the embedding layer of the MLP.
        """
        self.generator = torch.Generator().manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = context_length
        self.embedding_dimension = embedding_dimension
        self.neurons = neurons
        self.hidden_layers = hidden_layers
        self.model = nn.Sequential()

        self._prepare()

        if torch.cuda.is_available():
            self.model.cuda()

    def _prepare(self):
        """Builds the model, fetches the data and splits it.
        """
        # Prepare data.
        ingredients = encode_ingredient_lists(get_ingredient_lists())
        data = torch.tensor(create_ngram(ingredients, self.context_length + 1))
        train_length = int(0.8 * len(data))
        valid_length = int(0.1 * len(data))
        test_length = len(data) - train_length - valid_length

        train_set, valid_set, test_set = torch.utils.data.random_split(data, [train_length, valid_length, test_length], self.generator)
        self.train_set, self.valid_set, self.test_set = train_set.dataset[train_set.indices], valid_set.dataset[valid_set.indices], test_set.dataset[test_set.indices]

        # Create layers.
        self.model.add_module("embed", nn.Embedding(len(ingredient_to_code), self.embedding_dimension))
        self.model.add_module("flatten", nn.Flatten())

        self.model.add_module("linear_in", nn.Linear(self.context_length * self.embedding_dimension, self.neurons, bias=False))
        self.model.add_module("batch_in", nn.BatchNorm1d(self.neurons))
        self.model.add_module("activation_in", nn.ReLU())

        for i in range(self.hidden_layers):
            self.model.add_module(f"hidden_linear_{i}", nn.Linear(self.neurons, self.neurons, bias=False))
            self.model.add_module(f"hidden_batch_{i}", nn.BatchNorm1d(self.neurons))
            self.model.add_module(f"hidden_activation_{i}", nn.ReLU())
        
        self.model.add_module("linear_out", nn.Linear(self.neurons, len(ingredient_to_code)))

    def train(self, max_epochs: int = 20, batch_size: int = 4096) -> List[List[float]]:
        """Trains the model for as long as the validation loss decreases.

        Args:
            max_epochs (int, optional): The maximum amount of epochs to train for. The model will abort if the validation loss gain gets too small.
            batch_size (int, optional): The amount of data points to evaluate at once. Defaults to 4096 (fits in roughly 4GB of VRAM).

        Returns:
            List[List[float]]: The training and validation losses.
        """
        self.model.train()

        sampler = torch.utils.data.DataLoader(self.train_set, batch_size, shuffle=True, generator=self.generator)
        batches = len(sampler)
    
        optimizer = torch.optim.AdamW(self.model.parameters())
        losses = [[],[]]

        epoch = 1
        last_state_dict = {}

        while True:
            avg_loss = 0
            for batch in tqdm.tqdm(sampler, total=len(sampler), desc=f"Epoch {epoch}"):
                optimizer.zero_grad()

                x = batch[:, :-1].to(self.device)
                y = batch[:, -1].to(self.device)

                logits = self.model.forward(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                avg_loss += loss.item() / batches

                optimizer.step()

            losses[0].append(avg_loss)
            losses[1].append(self.test(valid_set=True, batch_size=batch_size))
            print(f"Average training loss of current epoch: {avg_loss}")
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
    
    def test(self, valid_set: bool = False, batch_size: int = 4096) -> float:
        """Tests the current model on the specified dataset.

        Args:
            valid_set (bool, optional): Whether to use the validation set instead of the test set. Defaults to False.
            batch_size (int, optional): The amount of data points to evaluate at once. Defaults to 4096 (fits in roughly 4GB of VRAM).

        Returns:
            float: The cross entropy loss.
        """
        self.model.eval()

        sampler = torch.utils.data.DataLoader(self.valid_set if valid_set else self.test_set, batch_size, shuffle=True, generator=self.generator)
        batches = len(sampler)

        avg_loss = 0
        for batch in sampler:
            x = batch[:, :-1].to(self.device)
            y = batch[:, -1].to(self.device)

            logits = self.model.forward(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            avg_loss += loss.item() / batches

        return avg_loss

    def generate_recipe_ingredients(self) -> List[str]:
        """Generates ingredients for a recipe. Here are a few samples:

        flour, baking powder, baking soda, salt, banana, butter, brown sugar, egg, vanilla

        ground beef, ground pork, onion, garlic clove, crushed tomato, egg, fresh parsley, salt & freshly ground black pepper, romano cheese

        green cabbage, onion, garlic clove, ham hock, beef broth, parsley, fresh cilantro, salt, pepper, salsa, paprika, chili powder

        Returns:
            List[str]: The ingredients.
        """
        self.model.eval()

        ingredients = []
        context = [0] * self.context_length

        while True:
            logits = self.model.forward(torch.tensor([context]).to(self.device))
            probs = torch.nn.functional.softmax(logits, dim=1)
            while True:
                # Loop until new ingredient.
                ingredient_code = torch.multinomial(probs, num_samples=1).item()
                if ingredient_code not in ingredients:
                    break

            if ingredient_code == 0:
                break

            context = context[1:] + [ingredient_code]
            ingredients.append(ingredient_code)
        
        return [code_to_ingredient[i] for i in ingredients]

    def save_model(self):
        torch.save(self.model.state_dict(), "./Models/MLP.pkl")

    def load_model(self):
        state_dict = torch.load("./Models/MLP.pkl")
        self.model.load_state_dict(state_dict)
