import Models.Instructions.RNNTorch
import Models.Instructions.RNN
import Models.Instructions.LSTMTorch
import Models.Instructions.LSTM
import Models.Instructions.Transformer
import Models.Instructions.GRU
import Models.Instructions.GRUTorch
import Models.Trainer
from Data import data
import torch

# Installing pytorch with CUDA is weird, check https://pytorch.org/ for instructions.

if __name__ == "__main__":
    model_type = input("Please choose a model type (Transformer, RNNTorch, RNN, LSTMTorch, LSTM, GRUTorch, GRU): ")
    context_length = 320
    torch.manual_seed(42)

    if model_type == "Transformer":
        trainer = Models.Trainer.Trainer(Models.Instructions.Transformer.Transformer(context_length, 2, 500, 700, 7, 0.0, data.enc.n_vocab + 1), context_length=context_length)
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
        print(f"Evaluating test error...")
        trainer.load_data("PoetryFoundationData.csv", context_length=context_length, size=1000)
        print(f"Test loss is {trainer.test(trainer.test_set[:100], show_progress=True)}")

        print("Model is ready. Press Enter for a new sample.")
        while True:
            temperature = input("Please provide a temperature, the default is 1.0: ")
            if temperature == "":
                temperature = 1.0
            top_k = input("Please provide a top_k, the default is 10: ")
            if top_k == "":
                top_k = 10
            top_p = input("Please provide a top_p, the default is 0.95: ")
            if top_p == "":
                top_p = 0.95

            ingredients = input("If you want, you can provide a beginning for the text generation now: ")

            trainer.generate_text(ingredients, temperature=float(temperature), top_k=int(top_k), top_p=float(top_p))
    else:
        trainer.load_data("PoetryFoundationData.csv", context_length=context_length, size=10000)
        print(trainer.train())
        trainer.save_checkpoint()
