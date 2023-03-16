from typing import *
import os
import shutil
import pandas
import json
import inflect
import csv
import requests


class Instruction:
    def __init__(self, instruction: str):
        self.instruction: str = instruction

    def __str__(self) -> str:
        return self.instruction.replace(",", " -")

class Ingredient:
    def __init__(self, name: str, amount: str, unit: str):
        self.name: str = name
        self.amount: str = amount
        self.unit: str = unit

    def __str__(self):
        return f"{' '.join([self.amount, self.unit])}:{self.name}".replace(",", "")

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
        ingredients_string = f"[{','.join([str(ingredient) for ingredient in self.ingredients])}]"
        instructions_string = f"[{','.join([str(instruction) for instruction in self.instructions])}]"
        return f"{self.name.replace(',', '')},{ingredients_string},{instructions_string}"


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
        i = (start_index // 10) + 1
        while recipes_to_retrieve == -1 or recipes_retrieved < recipes_to_retrieve:
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
                        for ingredient in recipe_object["recipeIngredient"][:]:
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

                        recipes_file.write(F"{start_index + recipes_retrieved},{current_recipe}\n")

                    except Exception as e:
                        print(f"Fetching recipe failed: {e}\n{recipe}")
                    
                    recipes_retrieved += 1
                    if recipes_retrieved >= recipes_to_retrieve:
                        break
            except Exception as e:
                print(f"Fetching page {i} failed: {e}")

            if not page_object["hasMore"]:
                break



def _download_recipes():
    """Downloads and extracts the food.com recipe dataset, if not already present.

    Returns:
        bool: Whether the data is now available or not.
    """
    if not os.path.exists("./Data/RAW_recipes.csv"):
        try:
            print("Downloading dataset...", end="")
            import kaggle
            
            if not os.path.exists("RAW_recipes.csv.zip"):
                # Download dataset.
                kaggle.api.dataset_download_file('shuyangli94/food-com-recipes-and-user-interactions', "RAW_recipes.csv", path='./')

                print(" Done")

            print("Extracting dataset...", end="")

            shutil.unpack_archive("RAW_recipes.csv.zip", "./Data")
            os.remove("RAW_recipes.csv.zip")

            print(" Done")
        except Exception as e:
            if "Could not find kaggle.json" in str(e):
                print("Kaggle credentials not found. Please follow the instructions here: https://github.com/Kaggle/kaggle-api#api-credentials")
            else:
                print(e)
    
    if not os.path.exists("./Data/RAW_recipes.csv"):
        exit(1)


_download_recipes()
ingredient_to_code: Dict[str, int] = {}
code_to_ingredient: Dict[int, str] = {}


def get_ingredient_lists() -> List[List[str]]:
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

    ingredient_set: Set[str] = set()
    for ingredient_list in ingredient_lists:
        for ingredient in ingredient_list:
            ingredient_set.add(ingredient)

    for i, ingredient in enumerate(ingredient_set):
        ingredient_to_code[ingredient] = i
        code_to_ingredient[i] = ingredient
    max_code = i

    return ingredient_lists


def encode_ingredient_lists(ingredient_lists: List[List[str]]) -> List[List[int]]:
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
            encoded_ingredient_lists[-1].append(ingredient_to_code[ingredient])

    return encoded_ingredient_lists


def get_ingredient_counts(ingredient_lists: List[List[Any]]) -> List[Tuple[Any, int]]:
    counts_dict: Dict[Any, int] = {}
    flat_list: List[Any] = [item for sublist in ingredient_lists for item in sublist]

    for ingredient in flat_list:
        if not ingredient in counts_dict:
            counts_dict[ingredient] = 1
        else:
            counts_dict[ingredient] += 1
    
    return sorted(list(counts_dict.items()), key=lambda x: x[-1], reverse=True)
