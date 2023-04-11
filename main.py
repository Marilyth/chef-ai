import Models.Instructions.RNNTorch
import Models.Instructions.Transformer
import os
import sys
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


if __name__ == "__main__":
    model_type = "RNNTorch"

    if model_type == "Transformer":
        model = Models.Instructions.Transformer.TransformerTrainer(200, 3, 500, 400, 4, 0.0)
    elif model_type == "RNNTorch":
        model = Models.Instructions.RNNTorch.RNNTrainer(500, 200, 1)

    if "test" in sys.argv:
        model.load_model()
        print("Model is ready. Press Enter for a new sample.")
        while True:
            ingredients = input("If you want, you can provide a comma seperated list of ingredients now: ")

            model.generate_recipe(ingredients)
    else:
        print(model.train())
        model.save_model()
