import Models.Instructions.RNNTorch
import Models.Instructions.RNN
import Models.Instructions.LSTMTorch
import Models.Instructions.LSTM
import Models.Instructions.Transformer
import Models.Instructions.GRU
import Models.Instructions.GRUTorch
import Models.Instructions.EncoderDecoder
import Models.Instructions.EncoderDecoderTorch
import Models.Instructions.FineTunedT5
from Data import data
import torch
import pytorch_lightning as lightning
import pytorch_lightning.callbacks as callbacks
import os

# Installing pytorch with CUDA is weird, check https://pytorch.org/ for instructions.

if __name__ == "__main__":
    model_type = input("Please choose a model type (Transformer, RNNTorch, RNN, LSTMTorch, LSTM, GRUTorch, GRU, EncoderDecoder, EncoderDecoderTorch, T5): ")
    encoder_length = 512
    context_length = 128
    torch.manual_seed(42)
    lightning.seed_everything(42)

    if model_type == "Transformer":
        model = Models.Instructions.Transformer.Transformer(context_length, 2, 500, 700, 7, 0.0, data.tokenizer.vocab_size)
    elif model_type == "RNNTorch":
        model = Models.Instructions.RNNTorch.RNNTorch(500, 500, data.tokenizer.vocab_size, 1)
    elif model_type == "RNN":
        model = Models.Instructions.RNN.RNN(500, 500, data.tokenizer.vocab_size, 1)
    elif model_type == "LSTM":
        model = Models.Instructions.LSTM.LSTM(1000, 500, data.tokenizer.vocab_size, 1)
    elif model_type == "LSTMTorch":
        model = Models.Instructions.LSTMTorch.LSTMTorch(1000, 1000, data.tokenizer.vocab_size, 1)
    elif model_type == "GRU":
        model = Models.Instructions.GRU.GRU(1000, 500, data.tokenizer.vocab_size, 1)
    elif model_type == "GRUTorch":
        model = Models.Instructions.GRUTorch.GRUTorch(1000, 1000, data.tokenizer.vocab_size, 1)
    elif model_type == "EncoderDecoder":
        model = Models.Instructions.EncoderDecoder.EncoderDecoderTransformer(encoder_length, context_length, 4, 500, 700, 2, 0.0, data.tokenizer.vocab_size)
    elif model_type == "EncoderDecoderTorch":
        model = Models.Instructions.EncoderDecoderTorch.EncoderDecoderTransformerTorch(encoder_length, context_length, 4, 2048, 512, 2, 0.0, data.tokenizer.vocab_size)
    elif model_type == "T5":
        model = Models.Instructions.FineTunedT5.FineTunedT5()

    mode = input("Please choose a mode (train, test, optuna): ")
    if mode == "test":
        model = model.load_from_checkpoint("checkpoints/" + type(model).__name__ + ".ckpt")

        while True:
            try:
                temperature = input("Please provide a temperature, the default is 1.0: ")
                if temperature == "":
                    temperature = 1.0
                top_k = input("Please provide a top_k, the default is -1: ")
                if top_k == "":
                    top_k = -1
                top_p = input("Please provide a top_p, the default is 1: ")
                if top_p == "":
                    top_p = 1
                compression = input("Please provide a compression value, the default is 0: ")
                if compression == "":
                    compression = 0

                # Take multiline input until the user enters eof.
                generation_input = input("If you want, you can provide a beginning for the text generation now: ")
                while True:
                    new_input = input()
                    if new_input.lower() == "eof":
                        break

                    generation_input += "\n" + new_input

                model.generate(generation_input, temperature=float(temperature), top_k=int(top_k), top_p=float(top_p), compression=int(compression), print_live=True)
            except Exception as e:
                continue
            
    elif mode == "train":
        # Create a trainer and a checkpointer.
        checkpointer = callbacks.ModelCheckpoint(monitor="val_loss", mode="min", dirpath="checkpoints", filename=type(model).__name__)
        trainer = lightning.Trainer(deterministic=True, callbacks=[checkpointer], val_check_interval=500)

        # Load datasets and split them into train, validation and test sets.
        dataset = data.SummarizationDataset(encoder_input_length=encoder_length, decoder_input_length=context_length, samples=250000, batch_size=4)

        # Load checkpoint if wanted and the file exists.
        if os.path.isfile("checkpoints/" + type(model).__name__ + ".ckpt") and input("Do you want to load a checkpoint? (y/n): ") == "y":
            trainer.fit(model, dataset, ckpt_path="checkpoints/" + type(model).__name__ + ".ckpt")
        else:
            trainer.fit(model, dataset)
        
        print(trainer.test(model, dataset))
    elif mode == "optuna":
        # Load datasets and split them into train, validation and test sets.
        dataset = data.TranslationDataset(encoder_input_length=encoder_length, decoder_input_length=context_length)
        dataset.setup("fit")
        # Optimize the model.
        print(model.optuna_optimize(dataset.train_dataloader(), dataset.val_dataloader(), n_trials=100))
    else:
        print("Invalid mode.")
