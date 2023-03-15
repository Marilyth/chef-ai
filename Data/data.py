from typing import *
import os
import shutil


class Instruction:
    def __init__(self, instruction: str):
        self.instruction: str = instruction


class Ingredient:
    def __init__(self, name: str, amount: float, unit: str):
        self.name: str = name
        self.amount: int = amount
        self.unit: str = unit


class Recipe:
    def __init__(self, name: str):
        self.name = name
        self.ingredients: List[Ingredient] = []
        self.instructions: List[Instruction] = []

    def add_ingredient(self, ingredient: Ingredient):
        self.ingredients.append(ingredient)
    
    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)


def get_recipes() -> bool:
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

            return False
    
    return os.path.exists("./Data/RAW_recipes.csv")
        