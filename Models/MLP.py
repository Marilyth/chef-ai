from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import get_ingredient_lists, encode_ingredient_lists, ingredient_to_code
import tqdm


def create_ngram(corpus: List[List[Any]], n: int = 2) -> Tuple[List[List[Any]]]:
    """Splits the corpus into a dataset with a context length of n.

    Args:
        corpus (List[List[Any]]): The original dataset.
        n (int, optional): The context length. Defaults to 2.

    Returns:
        Tuple[List[List[Any]]]: The dataset.
    """
    data = []

    for word in corpus:
        context = [0] * n
        for segment in word + [0]:
            context = context[1:] + [segment]
            data.append(context)
    
    return torch.tensor(data)


def create_mlp(context_length = 2):
    ingredients = encode_ingredient_lists(get_ingredient_lists())
    data = create_ngram(ingredients[:10000])
    train_length = int(0.8 * len(data))
    valid_length = int(0.1 * len(data))
    test_length = len(data) - train_length - valid_length

    train_set, valid_set, test_set = torch.utils.data.random_split(data, [train_length, valid_length, test_length], torch.Generator().manual_seed(42))

    # Create a NN with context_length features in, and amount of ingredients features out.
    mlp = nn.Sequential(
            nn.Linear(context_length - 1, 15),
            nn.BatchNorm1d(15),
            nn.ReLU(),

            nn.Linear(15, len(ingredient_to_code)),
            nn.BatchNorm1d(len(ingredient_to_code)),
            nn.ReLU(),

            nn.Softmax()
    )

    mlp.train(True)

    epochs = 4
    sampler = torch.utils.data.DataLoader(train_set, 8)
    #loader = torch.utils.data.DataLoader(sampler)
    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=2e-5, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-4, 1, len(sampler) * epochs)

    for epoch in range(epochs):
        batches = len(sampler)

        for batch in tqdm.tqdm(sampler, total=len(sampler), desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()

            x = batch[:, 0]
            y = batch[:, 1]

            logits = mlp.forward(x)
            loss = loss_fn.forward(logits, y)
            loss.backward()

            optimizer.step()
            scheduler.step()

