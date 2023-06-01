from typing import *
import os
import shutil
import pandas
import json
import re
import inflect
import csv
import requests
from tqdm import tqdm
from datasets import load_dataset, Dataset
import torch
import torch.utils.data
import pytorch_lightning as lightning
from transformers import AutoTokenizer, PreTrainedTokenizer


class EncoderDecoderDataset(torch.utils.data.Dataset):
    def __init__(self, encoder_inputs, decoder_inputs):
        self.encoder_inputs = torch.tensor(encoder_inputs)
        self.decoder_inputs = torch.tensor(decoder_inputs)

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        encoder_input = self.encoder_inputs[idx]
        decoder_input = self.decoder_inputs[idx]
        return encoder_input, decoder_input


class SummarizationDataset(lightning.LightningDataModule):
    def __init__(self, prefixes: List[str] = [None, None], train_batch_size: int = 8, test_batch_size: int = 8, encoder_input_length: int = 512, decoder_input_length: int = 128, samples: int = 2000, **kwargs):
        super().__init__()
        self.batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_source_length = encoder_input_length
        # +1 because the decoder is used as both input and output, shifted by 1.
        self.max_target_length = decoder_input_length + 1
        self.samples = samples
        self.prefixes = prefixes
        self.save_hyperparameters()

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train").remove_columns(["id"])[:self.samples]
            tokenized = self.tokenize_iteratively(self.train_dataset, [self.max_source_length - 1, self.max_target_length - 1], ["article", "highlights"])
            self.train_dataset = EncoderDecoderDataset(tokenized[0], tokenized[1])
            
            self.val_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation").remove_columns(["id"])[:1000]
            tokenized = self.tokenize_iteratively(self.val_dataset, [self.max_source_length - 1, self.max_target_length - 1], ["article", "highlights"])
            self.val_dataset = EncoderDecoderDataset(tokenized[0], tokenized[1])
        elif stage == "test":
            self.test_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test").remove_columns(["id"])[:1000]
            tokenized = self.tokenize_iteratively(self.test_dataset, [self.max_source_length - 1, self.max_target_length - 1], ["article", "highlights"])
            self.test_dataset = EncoderDecoderDataset(tokenized[0], tokenized[1])
    
    def tokenize_iteratively(self, dataset: Dataset, max_lengths: List[int], columns: List[str], step_size: int = 1000, column_first: bool = True, **kwargs) -> List[List[int]]:
        """Tokenizes a dataset in steps of step_size to avoid memory errors. Returns a list of tokenized columns.

        Args:
            dataset (Dataset): The dataset to tokenize.
            max_lengths (List[int]): The maximum lengths of the tokenized columns.
            columns (List[str]): The columns to tokenize.
            step_size (int, optional): The step size to tokenize the dataset in. Defaults to 1000.
            column_first (bool, optional): Whether the columns are the first dimension of the dataset. Defaults to True.
        Returns:
            List[List[int]]: The tokenized columns.
        """
        tokenized = []
        for column in columns:
            tokenized.append([])

        for i in tqdm(range(0, len(dataset[columns[0]] if column_first else dataset), step_size), desc="Tokenizing"):
            for j, column in enumerate(columns):
                if column_first:
                    data = dataset[column][i:i+step_size]
                else:
                    data = [data_point[column] for data_point in dataset[i:i+step_size]]

                prefix = self.prefixes[j]
                if prefix is not None:
                    data = [prefix + data_point for data_point in data]

                new_tokens = add_start_tokens(tokenizer(data, max_length=max_lengths[j], padding="max_length", truncation=True, **kwargs).data["input_ids"])
                tokenized[j].extend(new_tokens)

        return tokenized

    def encode(self, text: List[str]) -> List[List[int]]:
        return encode_texts(text)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
    
class TranslationDataset(SummarizationDataset):
    def __init__(self, prefixes: List[str] = [None, None], train_batch_size: int = 8, test_batch_size: int = 8, encoder_input_length: int = 64, decoder_input_length: int = 64, samples: int = 2000, **kwargs):
        super().__init__(train_batch_size=train_batch_size, test_batch_size=test_batch_size, prefixes=prefixes, encoder_input_length=encoder_input_length, decoder_input_length=decoder_input_length, samples=samples, **kwargs)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = load_dataset("opus100", "de-en", split="train")[:self.samples]["translation"]
            tokenized = self.tokenize_iteratively(self.train_dataset, [self.max_source_length - 1, self.max_target_length - 1], ["de", "en"], column_first=False)
            self.train_dataset = EncoderDecoderDataset(tokenized[0], tokenized[1])
            
            self.val_dataset = load_dataset("opus100", "de-en", split="validation")[:1000]["translation"]
            tokenized = self.tokenize_iteratively(self.val_dataset, [self.max_source_length - 1, self.max_target_length - 1], ["de", "en"], column_first=False)
            self.val_dataset = EncoderDecoderDataset(tokenized[0], tokenized[1])
        elif stage == "test":
            self.test_dataset = load_dataset("opus100", "de-en", split="test")[:1000]["translation"]
            tokenized = self.tokenize_iteratively(self.test_dataset, [self.max_source_length - 1, self.max_target_length - 1], ["de", "en"], column_first=False)
            self.test_dataset = EncoderDecoderDataset(tokenized[0], tokenized[1])

class PubMedDataset(SummarizationDataset):
    def __init__(self, prefixes: List[str] = [None, None], train_batch_size: int = 8, test_batch_size: int = 8, encoder_input_length: int = 64, decoder_input_length: int = 64, samples: int = 2000, **kwargs):
        super().__init__(train_batch_size=train_batch_size, test_batch_size=test_batch_size, prefixes=prefixes, encoder_input_length=encoder_input_length, decoder_input_length=decoder_input_length, samples=samples, **kwargs)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = load_dataset("scientific_papers", "pubmed", split="train")[:self.samples]
            tokenized = self.tokenize_iteratively(self.train_dataset, [self.max_source_length - 1, self.max_target_length - 1], ["article", "abstract"])
            self.train_dataset = EncoderDecoderDataset(tokenized[0], tokenized[1])
            
            self.val_dataset = load_dataset("scientific_papers", "pubmed", split="validation")[:80]
            tokenized = self.tokenize_iteratively(self.val_dataset, [self.max_source_length - 1, self.max_target_length - 1], ["article", "abstract"])
            self.val_dataset = EncoderDecoderDataset(tokenized[0], tokenized[1])
        elif stage == "test":
            self.test_dataset = load_dataset("scientific_papers", "pubmed", split="test")[:80]
            tokenized = self.tokenize_iteratively(self.test_dataset, [self.max_source_length - 1, self.max_target_length - 1], ["article", "abstract"])
            self.test_dataset = EncoderDecoderDataset(tokenized[0], tokenized[1])

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-large")
#tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
#tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
#tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")
#tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")

def get_texts(data_name: str = "RAW_recipes.csv", **kwargs) -> pandas.DataFrame:
    """Returns a list of all elements in the dataset of the file.

    Args:
        file_name (str, optional): The name of the file to read from. Defaults to "RAW_recipes.csv".
        sample_size (Optional[int], optional): The number of elements to sample. Defaults to None.
        random_seed (Optional[int], optional): The random seed to use. Defaults to None. If None, the sample is not random.
    Returns:
        List[str]: The list of elements.
    """
    texts = []

    # Load datasets from https://huggingface.co/datasets?sort=likes
    texts = load_dataset(data_name, **kwargs).shuffle(seed=42)

    return texts.data["train"].to_pandas()


def encode_texts(recipes: List[str]) -> List[List[int]]:
    return [tokenizer.encode(recipe) for recipe in recipes]


def decode_texts(recipes: List[List[int]]) -> List[str]:
    return [tokenizer.decode(recipe) for recipe in recipes]


def get_word_counts(ingredient_lists: List[List[Any]]) -> List[Tuple[Any, int]]:
    counts_dict: Dict[Any, int] = {}
    flat_list: List[Any] = [item for sublist in ingredient_lists for item in sublist]

    for ingredient in flat_list:
        if not ingredient in counts_dict:
            counts_dict[ingredient] = 1
        else:
            counts_dict[ingredient] += 1
    
    return sorted(list(counts_dict.items()), key=lambda x: x[-1], reverse=True)


def create_ngram(corpus: List[List[Any]], n: int = 2, pad_code: int = 50259, add_ending: bool = True) -> Tuple[List[List[Any]]]:
    """Splits the corpus into a dataset with a context length of n.

    Args:
        corpus (List[List[Any]]): The original dataset.
        n (int, optional): The context length. Defaults to 2.
        pad_code (int, optional): The encoding for the padding character. Defaults to 0.
        add_ending (bool, optional): Whether to include a padding in the end. Defaults to True.

    Returns:
        Tuple[List[List[Any]]]: The dataset.
    """
    data = []

    for word in corpus:
        context = [pad_code] * n
        for segment in (word + [0]) if add_ending else word:
            context = context[1:] + [segment]
            data.append(context)
    
    return data


def create_sliding(corpus: List[List[Any]], n: int = 2, pad_code: int = 50259, remove_excess: bool = False) -> Tuple[List[List[Any]]]:
    """Creates a sliding window of maximum size n, beginning with 1 pad. 
    Useful for transformers who can learn on any context length.
    If the text does not contain n words, it keeps its size + 1 (left pad).
    ["i", "am", "a", "test"], n=100
    ["pad", "i", "am", "a", "test"]

    Args:
        corpus (List[List[Any]]): The original dataset.
        n (int, optional): The maximum window size. Defaults to 2.
        pad_code (int, optional): The encoding for the padding character. Defaults to 0.
        remove_excess (bool, optional): Whether to only return 1 window, and remove the excess. Defaults to False.

    Returns:
        Tuple[List[List[Any]]]: The dataset.
    """
    data = []

    for word in corpus:
        context = [pad_code] + word
        for i in range(max((len(word) + 1) - n, 1)):
            data.append(context[i:i + n])
            if remove_excess:
                break
    
    return data


def chunk_text(text: str, max_chunk_length: int = 256) -> List[str]:
    """Chunks a text into smaller pieces, so that it fits into the transformer's input size.
    The text is split at the end of a sentence, so that the chunks are still readable.
    Iterating over chunks is a simple way to process large texts.

    Args:
        text (str): The text to chunk.
        max_chunk_length (int, optional): The maximum words per chunk. Defaults to 256.

    Returns:
        List[str]: The list of chunks.
    """
    chunks = []
    chunk = ""

    for sentence in re.split("([^.?!]+[.?!]+)", text):
        word_count = (chunk + sentence).count(" ") + 1
        if word_count > max_chunk_length:
            chunks.append(chunk.strip())
            chunk = ""
        chunk += sentence

    chunks.append(chunk)
    return chunks


def add_start_tokens(text: List[int], start_tokens: List[int] = [tokenizer.pad_token_id]) -> List[int]:
    return [[tokenizer.pad_token_id] + t for t in text]


def split(data: List[Any], train_ratio: float = 0.8, valid_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[List[Any], List[Any], List[Any]]:
    """Splits the dataset into train, validation and test sets.

    Args:
        data (List[Any]): The dataset.
        train_ratio (float, optional): The ratio of the train set. Defaults to 0.8.
        valid_ratio (float, optional): The ratio of the validation set. Defaults to 0.1.
        test_ratio (float, optional): The ratio of the test set. Defaults to 0.1.

    Returns:
        Tuple[List[Any], List[Any], List[Any]]: The train, validation and test sets.
    """
    train_length = int(train_ratio * len(data))
    valid_length = int(valid_ratio * len(data))
    test_length = len(data) - train_length - valid_length

    train_set, valid_set, test_set = torch.utils.data.random_split(data, [train_length, valid_length, test_length], generator=torch.Generator().manual_seed(42))
    return [train_set.dataset[i] for i in train_set.indices],\
            [valid_set.dataset[i] for i in valid_set.indices],\
            [test_set.dataset[i] for i in test_set.indices]


def list_to_dataloader(data: List[Any], batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """Converts a list to a dataloader.

    Args:
        data (List[Any]): The dataset.
        batch_size (int, optional): The batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        length (int, optional): The length of the sequences. Defaults to 320.
    Returns:
        torch.utils.data.DataLoader: The dataloader.
    """
    # Transform the dataset into a Tensor of tensors of equal length.
    data_tensors = [ torch.tensor(t) for t in data ]
    data_tensors = torch.nn.utils.rnn.pad_sequence(data_tensors, batch_first=True)

    return torch.utils.data.DataLoader(data_tensors, batch_size=batch_size, shuffle=shuffle)
    