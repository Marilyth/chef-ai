import Models.Instructions.RNNTorch
import Models.Instructions.RNN
import Models.Instructions.LSTMTorch
import Models.Instructions.LSTM
import Models.Instructions.Transformer
import Models.Instructions.GRU
import Models.Instructions.GRUTorch
import Models.Trainer
from Data import data
import os
import sys
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


if __name__ == "__main__":
    model_type = input("Please choose a model type (Transformer, RNNTorch, RNN, LSTMTorch, LSTM, GRUTorch, GRU): ")

    if model_type == "Transformer":
        trainer = Models.Trainer.Trainer(Models.Instructions.Transformer.Transformer(200, 3, 500, 400, 4, 0.0, data.enc.n_vocab + 1), context_length=200)
    elif model_type == "RNNTorch":
        trainer = Models.Trainer.Trainer(Models.Instructions.RNNTorch.RNNTorch(500, 500, data.enc.n_vocab + 1, 1))
    elif model_type == "RNN":
        trainer = Models.Trainer.Trainer(Models.Instructions.RNN.RNN(500, 500, data.enc.n_vocab + 1, 1))
    elif model_type == "LSTM":
        trainer = Models.Trainer.Trainer(Models.Instructions.LSTM.LSTM(1000, 500, data.enc.n_vocab + 1, 1))
    elif model_type == "LSTMTorch":
        trainer = Models.Trainer.Trainer(Models.Instructions.LSTMTorch.LSTMTorch(1000, 1000, data.enc.n_vocab + 1, 1))
    elif model_type == "GRU":
        trainer = Models.Trainer.Trainer(Models.Instructions.GRU.GRU(1000, 500, data.enc.n_vocab + 1, 1))
    elif model_type == "GRUTorch":
        trainer = Models.Trainer.Trainer(Models.Instructions.GRUTorch.GRUTorch(1000, 1000, data.enc.n_vocab + 1, 1))

    mode = input("Please choose a mode (train, test): ")
    if mode == "test":
        trainer.load_model()
        print(f"Evaluating test error...")
        trainer.load_data("PoetryFoundationData.csv", context_length=200)
        print(f"Test loss is {trainer.test(trainer.test_set[:100], show_progress=True)}")

        print("Model is ready. Press Enter for a new sample.")
        while True:
            ingredients = input("If you want, you can provide a beginning for the text generation now: ")

            trainer.generate_text(ingredients)
    else:
        trainer.load_data("PoetryFoundationData.csv", context_length=200)
        print(trainer.train())
        trainer.save_model()
