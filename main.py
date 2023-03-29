import Models.Ingredients.Transformer
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


if __name__ == "__main__":
    transformer = Models.Ingredients.Transformer.TransformerTrainer(10, 6, 348, 348, 6, 0.0)
    transformer.load_model()
    #print(transformer.train(1))
    #transformer.save_model()

    print("Model is ready. Enter for a new row of ingredients.")
    while True:
        input()
        print(", ".join(transformer.generate_recipe_ingredients()))
