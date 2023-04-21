import Models.Instructions.RNNTorch
import Models.Instructions.RNN
import Models.Instructions.LSTMTorch
import Models.Instructions.LSTM
import Models.Instructions.Transformer
import Models.Instructions.GRU
import Models.Instructions.GRUTorch
from Data import data
import torch
import pytorch_lightning as lightning
import pytorch_lightning.callbacks as callbacks
import os

# Installing pytorch with CUDA is weird, check https://pytorch.org/ for instructions.

if __name__ == "__main__":
    model_type = input("Please choose a model type (Transformer, RNNTorch, RNN, LSTMTorch, LSTM, GRUTorch, GRU): ")
    context_length = 320
    torch.manual_seed(42)
    lightning.seed_everything(42)

    if model_type == "Transformer":
        model = Models.Instructions.Transformer.Transformer(context_length, 2, 500, 700, 7, 0.0, data.enc.n_vocab + 1)
    elif model_type == "RNNTorch":
        model = Models.Instructions.RNNTorch.RNNTorch(500, 500, data.enc.n_vocab + 1, 1)
    elif model_type == "RNN":
        model = Models.Instructions.RNN.RNN(500, 500, data.enc.n_vocab + 1, 1)
    elif model_type == "LSTM":
        model = Models.Instructions.LSTM.LSTM(1000, 500, data.enc.n_vocab + 1, 1)
    elif model_type == "LSTMTorch":
        model = Models.Instructions.LSTMTorch.LSTMTorch(1000, 1000, data.enc.n_vocab + 1, 1)
    elif model_type == "GRU":
        model = Models.Instructions.GRU.GRU(1000, 500, data.enc.n_vocab + 1, 1)
    elif model_type == "GRUTorch":
        model = Models.Instructions.GRUTorch.GRUTorch(1000, 1000, data.enc.n_vocab + 1, 1)

    mode = input("Please choose a mode (train, test, optuna): ")
    if mode == "test":
        model = model.load_from_checkpoint("checkpoints/" + type(model).__name__ + ".ckpt")

        while True:
            temperature = input("Please provide a temperature, the default is 1.0: ")
            if temperature == "":
                temperature = 1.0
            top_k = input("Please provide a top_k, the default is -1: ")
            if top_k == "":
                top_k = -1
            top_p = input("Please provide a top_p, the default is 1: ")
            if top_p == "":
                top_p = 1

            ingredients = input("If you want, you can provide a beginning for the text generation now: ")
            print(model.generate(ingredients, temperature=float(temperature), top_k=int(top_k), top_p=float(top_p)))
    elif mode == "train":
        # Create a trainer and a checkpointer.
        checkpointer = callbacks.ModelCheckpoint(monitor="val_loss", mode="min", dirpath="checkpoints", filename=type(model).__name__)
        trainer = lightning.Trainer(deterministic=True, gradient_clip_val=0.5, val_check_interval=100, limit_val_batches=100, callbacks=[checkpointer])

        # Load datasets and split them into train, validation and test sets.
        dataset = data.get_texts("merve/poetry")["content"].values.tolist()
        dataset = data.encode_texts(dataset)
        train, valid, test = data.split(dataset, 0.8, 0.1, 0.1)
        train_loader = data.list_to_dataloader(data.create_sliding(train, context_length), 8)
        valid_loader = data.list_to_dataloader(data.create_sliding(valid, context_length), 8, shuffle=False)
        test_loader = data.list_to_dataloader(data.create_sliding(test, context_length), 8, shuffle=False)

        # Load checkpoint if wanted and the file exists.
        if os.path.isfile("checkpoints/" + type(model).__name__ + ".ckpt") and input("Do you want to load a checkpoint? (y/n): ") == "y":
            trainer.fit(model, train_loader, valid_loader, ckpt_path="checkpoints/" + type(model).__name__ + ".ckpt")
        else:
            trainer.fit(model, train_loader, valid_loader)
        
        print(trainer.test(model, test_loader))
    elif mode == "optuna":
        # Load datasets and split them into train, validation and test sets.
        dataset = data.get_texts("merve/poetry")["content"].values.tolist()
        dataset = data.encode_texts(dataset)
        train, valid, test = data.split(dataset, 0.8, 0.1, 0.1)
        train_loader = data.list_to_dataloader(data.create_sliding(train, context_length), 8)
        valid_loader = data.list_to_dataloader(data.create_sliding(valid, context_length), 8, shuffle=False)
        test_loader = data.list_to_dataloader(data.create_sliding(test, context_length), 8, shuffle=False)

        # Optimize the model.
        print(model.optuna_optimize(train_loader, valid_loader))
    else:
        print("Invalid mode.")
