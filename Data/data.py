from typing import *
import os
import shutil
import pandas
import json
import inflect
import csv
import requests
from tqdm import tqdm
import tiktoken
import torch
import re


class Instruction:
    def __init__(self, instruction: str):
        self.instruction: str = instruction

    def __str__(self) -> str:
        return self.instruction.replace(",", " -")

class Ingredient:
    def __init__(self, name: str, amount: str, unit: str):
        self.name: str = name
        self.amount: str = amount if amount else "1"
        self.unit: str = unit

    def __str__(self):
        return f"{' '.join([self.amount, self.unit]).strip()}:{self.name}".replace(",", "")

class Recipe:
    def __init__(self, name: str, stars: float, ratings: int):
        self.name = name
        self.stars = stars
        self.ratings = ratings
        self.ingredients: List[Ingredient] = []
        self.instructions: List[Instruction] = []

    def add_ingredient(self, ingredient: Ingredient):
        self.ingredients.append(ingredient)
    
    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)

    def __str__(self) -> str:
        ingredients_string = f"[{','.join([ingredient.name for ingredient in self.ingredients])}]"
        units_string = f"[{','.join([' '.join([ingredient.amount, ingredient.unit]).strip() for ingredient in self.ingredients])}]"
        instructions_string = f"[{','.join([str(instruction) for instruction in self.instructions])}]"
        return f"{self.name.replace(',', '')},{ingredients_string},{units_string},{instructions_string}"


def crawl_food_com(start_index: int = 0, recipes_to_retrieve: int = -1):
    """Crawls through https://www.food.com/recipe/all/trending?pn={page} and extracts recipe information.
    The recipes are written into recipes.csv.

    Args:
        recipes_to_retrieve (int, optional): The amount of recipes to retrieve. Defaults to -1 (until done).
        start_index (int, optional): The recipe to start at. Defaults to 0.
    """
    trending_page = "https://www.food.com/recipe/all/trending?pn={0}"

    with open("./Data/recipes.csv", "a+") as recipes_file:
        # Will break if no more recipes are found.
        recipes_retrieved = 0
        first_page = trending_page.format(1)
        response = requests.get(first_page).text
        total_recipes = int(json.loads(response.split("var initialData = ")[-1].split(";\n\n</script>")[0])["response"]["totalResultsCount"])
        end_index = (min(total_recipes, recipes_to_retrieve + start_index) if recipes_to_retrieve > 0 else total_recipes) - 1
        end_page_index = (end_index // 10) + 1
        start_page_index = (start_index // 10) + 1

        for i in tqdm(range(start_page_index, end_page_index + 1), f"Page progression"):
            try:
                current_page = trending_page.format(i)
                response = requests.get(current_page).text
                page_data = response.split("var initialData = ")[-1].split(";\n\n</script>")[0]
                page_object = json.loads(page_data)
                for recipe in page_object["response"]["results"]:
                    try:
                        current_recipe = Recipe(recipe["main_title"], float(recipe["main_rating"]), int(recipe["main_num_ratings"]))
                        recipe_details = requests.get(recipe["record_url"]).text
                        recipe_data = '{"@context"' + recipe_details.split('{"@context"')[-2].split("</script>")[0]
                        recipe_object = json.loads(recipe_data)

                        # Add ingredients.
                        for ingredient in recipe_object["recipeIngredient"][(start_index % 10) if i == start_page_index else 0:]:
                            # Always take first option.
                            ingredient = ingredient.split(" or ")[0]
                            ingredient_description = ingredient.split("   ")[-1]
                            ingredient_quantity = ingredient[:-len(ingredient_description)].strip()
                            ingredient_quantity =  ingredient_quantity.split("(")[0] + ingredient_quantity.split(")")[-1] if "(" in ingredient_quantity and ")" in ingredient_quantity else ingredient_quantity
                            ingredient_amount, ingredient_unit = ingredient_quantity.split("  ") if "  " in ingredient_quantity else [ingredient_quantity, ""]
                            ingredient_unit = ingredient_unit.replace(" ", "")
                            if not ingredient_unit:
                                ingredient_unit = "unit"

                            current_recipe.add_ingredient(Ingredient(ingredient_description.replace(",", ".").strip(), ingredient_amount.strip(), ingredient_unit.strip()))
                        
                        # Add instructions.
                        for instruction in recipe_object["recipeInstructions"]:
                            current_recipe.add_instruction(Instruction(instruction["text"].strip()))

                        recipes_file.write(f"{start_index + recipes_retrieved},{current_recipe}\n")
                    except Exception as e:
                        print(f"Fetching recipe failed: {e}\n{recipe}")
                    
                    recipes_retrieved += 1
                    if recipes_retrieved >= recipes_to_retrieve:
                        break
            except Exception as e:
                print(f"Fetching page {i} failed: {e}")

            if not page_object["hasMore"]:
                break



def _download_dataset(name="shuyangli94/food-com-recipes-and-user-interactions", file_name="RAW_recipes.csv"):
    """Downloads and extracts the food.com recipe dataset, if not already present.

    Returns:
        bool: Whether the data is now available or not.
    """
    if not os.path.exists(f"./Data/{file_name}"):
        try:
            print("Downloading dataset...", end="")
            import kaggle
            
            if not os.path.exists(file_name):
                # Download dataset.
                kaggle.api.dataset_download_file(name, file_name, path='./')

                print(" Done")

            print("Extracting dataset...", end="")

            shutil.unpack_archive(file_name + ".zip", "./Data")
            os.remove(file_name + ".zip")

            print(" Done")
        except Exception as e:
            if "Could not find kaggle.json" in str(e):
                print("Kaggle credentials not found. Please follow the instructions here: https://github.com/Kaggle/kaggle-api#api-credentials")
            else:
                print(e)
    
    if not os.path.exists(f"./Data/{file_name}"):
        exit(1)


#_download_recipes()
word_to_code: Dict[str, int] = {".": 0}
code_to_word: Dict[int, str] = {0: "."}
gpt_encoder = tiktoken.get_encoding("gpt2")
enc = tiktoken.Encoding(name="gpt2_recipe", 
                        pat_str=gpt_encoder._pat_str,
                        mergeable_ranks=gpt_encoder._mergeable_ranks,
                        special_tokens={
                            **gpt_encoder._special_tokens,
                            "<|ingredients_end|>": 50257,
                            "<|next_step|>": 50258,
                            "<|padding|>": 50259,
                        })

def get_text_lists() -> List[List[str]]:
    """Returns all recipe's ingredients.

    Returns:
        List[List[str]]: The ingredient lists.
    """
    if not os.path.exists("./Data/ingredients.csv"):
        singularizer = inflect.engine()
        data_frame = pandas.read_csv("./Data/RAW_recipes.csv")
        ingredient_lists = data_frame["ingredients"]

        # Parse list expression to actual list. Remove entries which contain ".
        ingredient_lists: List[List[str]] = [json.loads(ingredient_list.replace(" and ", "', '").replace("'","\"")) for ingredient_list in ingredient_lists.to_list() if "\"" not in ingredient_list]

        with open("./Data/ingredients.csv", "w+") as ingredients_file:
            for i, ingredient_list in enumerate(ingredient_lists[:]):
                for j, ingredient in enumerate(ingredient_list[:]):
                    singular_words = []

                    # Inflect returns False if already singular, else the singular word. This is so annoying.
                    for word in ingredient.split(" "):
                        if word:
                            singular_word = singularizer.singular_noun(word)
                            singular_words.append(singular_word if singular_word else word)

                    singular_ingredient = " ".join(singular_words)
                    ingredient_lists[i][j] = singular_ingredient

            ingredients_file.write("\n".join([",".join(ingredient_list) for ingredient_list in ingredient_lists]))
    else:
        with open("./Data/ingredients.csv", "r") as ingredients_file:
            ingredient_lists = list(csv.reader(ingredients_file, delimiter=","))
            ingredient_lists = [ingredient_list for ingredient_list in ingredient_lists]

    ingredient_set: Set[str] = set()
    for ingredient_list in ingredient_lists:
        for ingredient in ingredient_list:
            ingredient_set.add(ingredient)

    for i, ingredient in enumerate(sorted(list(ingredient_set))):
        word_to_code[ingredient] = i + 1
        code_to_word[i + 1] = ingredient

    return ingredient_lists


def get_texts(file_name: str = "RAW_recipes.csv", sample_size: Optional[int] = None, random_seed: Optional[int] = None) -> List[str]:
    """Returns a list of all elements in the dataset of the file.

    Args:
        file_name (str, optional): The name of the file to read from. Defaults to "RAW_recipes.csv".
        sample_size (Optional[int], optional): The number of elements to sample. Defaults to None.
        random_seed (Optional[int], optional): The random seed to use. Defaults to None. If None, the sample is not random.
    Returns:
        List[str]: The list of elements.
    """
    texts = []

    data_frame = pandas.read_csv(f"./Data/{file_name}")
    if sample_size:
        if random_seed:
            data_frame = data_frame.sample(sample_size, random_state=random_seed)
        else:
            data_frame = data_frame.sample(sample_size)

    if file_name == "RAW_recipes.csv":
        ingredient_lists = data_frame["ingredients"].to_list()
        instruction_lists = data_frame["steps"].to_list()

        for ingredients, instructions in zip(ingredient_lists, instruction_lists):
            recipe = ingredients.replace("['", "").replace("']", "").replace("', '", ", ").replace("', \"", ", ").replace("\", '", ", ").replace("\", \"", ", ") + "<|ingredients_end|>"
            recipe += instructions.replace("['", "").replace("']", "").replace("', '", "<|next_step|>").replace("', \"", "<|next_step|>").replace("\", '", "<|next_step|>").replace("\", \"", "<|next_step|>") + "<|endoftext|>"
            texts.append(recipe)
    elif file_name == "PoetryFoundationData.csv":
        texts: List[str] = data_frame["Poem"].to_list()
        for i in range(len(texts)):
            texts[i] = texts[i].strip().replace("\r\r\n", "<|next_step|>") + "<|endoftext|>"
            # Remove multiple spaces and replace them with a single one.
            texts[i] = re.sub("\s\s+", " ", texts[i])

    return texts


def encode_texts(recipes: List[str]) -> List[List[int]]:
    return enc.encode_batch(recipes, allowed_special="all")


def decode_texts(recipes: List[List[int]]) -> List[str]:
    return [enc.decode(recipe) for recipe in recipes]


def encode_text_lists(ingredient_lists: List[List[str]]) -> List[List[int]]:
    """Creates an encoded ingredient list, equivalent to the passed one.

    Args:
        ingredient_lists (List[List[str]]): The ingredient lists to encode.

    Returns:
        List[List[int]]: The encoded ingredient lists.
    """
    encoded_ingredient_lists: List[List[int]] = []

    for ingredient_list in ingredient_lists:
        encoded_ingredient_lists.append([])
        for ingredient in ingredient_list:
            encoded_ingredient_lists[-1].append(word_to_code[ingredient])

    return encoded_ingredient_lists


def get_word_counts(ingredient_lists: List[List[Any]]) -> List[Tuple[Any, int]]:
    counts_dict: Dict[Any, int] = {}
    flat_list: List[Any] = [item for sublist in ingredient_lists for item in sublist]

    for ingredient in flat_list:
        if not ingredient in counts_dict:
            counts_dict[ingredient] = 1
        else:
            counts_dict[ingredient] += 1
    
    return sorted(list(counts_dict.items()), key=lambda x: x[-1], reverse=True)


def create_ngram(corpus: List[List[Any]], n: int = 2, pad_code: int = 0, add_ending: bool = True) -> Tuple[List[List[Any]]]:
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


def create_sliding(corpus: List[List[Any]], n: int = 2, pad_code: int = 0) -> Tuple[List[List[Any]]]:
    """Creates a sliding window of maximum size n, beginning with 1 pad. 
    Useful for transformers who can learn on any context length.
    If the text does not contain n words, it keeps its size + 1 (left pad).
    ["i", "am", "a", "test"], n=100
    ["pad", "i", "am", "a", "test"]

    Args:
        corpus (List[List[Any]]): The original dataset.
        n (int, optional): The maximum window size. Defaults to 2.
        pad_code (int, optional): The encoding for the padding character. Defaults to 0.

    Returns:
        Tuple[List[List[Any]]]: The dataset.
    """
    data = []

    for word in corpus:
        context = [pad_code] + word
        for i in range(max((len(word) + 1) - n, 1)):
            data.append(context[i:i + n])
    
    return data
