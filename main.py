import Models.MLP
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


if __name__ == "__main__":
    mlp_model = Models.MLP.MLP(10, 1, 450, 350)
    losses = mlp_model.train()
    mlp_model.save_model()

    print("Model is ready. Enter for a new row of ingredients.")
    while True:
        input()
        print(", ".join(mlp_model.generate_recipe_ingredients()))
