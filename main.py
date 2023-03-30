import Models.Instructions.Transformer
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


if __name__ == "__main__":
    transformer = Models.Instructions.Transformer.TransformerTrainer(200, 3, 500, 400, 4, 0.0)
    #transformer.load_model()
    transformer.train(max_time=900)
    transformer.save_model()

    print("Model is ready. Press Enter for a new sample.")
    while True:
        ingredients = input("If you want, you can provide a comma seperated list of ingredients now: ")

        print(transformer.generate_recipe(ingredients))
