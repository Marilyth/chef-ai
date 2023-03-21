from Data.data import get_ingredient_lists, encode_ingredient_lists, get_ingredient_counts, crawl_food_com
import Models.MLP
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


if __name__ == "__main__":
    #crawl_food_com(0, 500)
    mlp_model = Models.MLP.MLP(10, 4, 450)
    #mlp_model.load_model()
    losses = mlp_model.train()
    mlp_model.save_model()

    print("Model is ready. Enter for a new row of ingredients.")
    while True:
        input()
        print(", ".join(mlp_model.generate_recipe_ingredients()))
