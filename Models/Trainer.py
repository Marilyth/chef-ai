from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time


class Trainer:
    def __init__(self, model: nn.Module, context_length: int = 1):
        """Initializes a trainer for the specified model.

        Args:
            model (nn.Module): The model to train.
            context_length (int): The context length the predict the next word with during generation. For RNN models this is 1.
        """
        self.generator = torch.Generator().manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = context_length

        self.model = model
        self.model.to(self.device)

    def _collate_fn_pad(self, batch):
        """Pad the batch to be of uniform length.
        """
        # Pad tensors to be of uniform length.
        batch = [ torch.Tensor(t) for t in batch ]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

        return batch

    def load_data(self, file_name: str="RAW_recipes.csv", size: int = 1000, context_length: int = -1):
        """Loads the data from the specified file. If context_length is -1, the context_length is derived from the Trainer.

        Args:
            file_name (str, optional): The file to load the data from. Defaults to "RAW_recipes.csv".
            size (int, optional): The number of texts to load. Defaults to 1000.
            context_length (int, optional): The context length to use during training. Defaults to -1.
        """
        if context_length == -1:
            context_length = self.context_length

        # Prepare data.
        texts = encode_texts(get_texts(file_name)[:size])

        # This data is of variable length. It needs to be packed before forward pass.
        data = [torch.tensor(datapoint) for datapoint in create_sliding(texts, context_length + 1, 50259)]

        train_length = int(0.8 * len(data))
        valid_length = int(0.1 * len(data))
        test_length = len(data) - train_length - valid_length

        train_set, valid_set, test_set = torch.utils.data.random_split(data, [train_length, valid_length, test_length], self.generator)
        self.train_set, self.valid_set, self.test_set = [train_set.dataset[i] for i in train_set.indices],\
                                                        [valid_set.dataset[i] for i in valid_set.indices],\
                                                        [test_set.dataset[i] for i in test_set.indices]

    def train(self, max_epochs: int = 20, max_time: int = -1, max_iterations: int = -1, batch_size: int = 8, progress_report: int = 100) -> List[List[float]]:
        """Trains the model for as long as the validation loss decreases.

        Args:
            max_epochs (int, optional): The maximum amount of epochs to train for. Defaults to 20.
            max_time (int, optional): The maximum amount of time to train for. Defaults to -1.
            max_iterations (int, optional): The maximum amount of iterations to train for. Defaults to -1.
            batch_size (int, optional): The batch size. Defaults to 8.
            progress_report (int, optional): The amount of iterations between progress reports. Defaults to 100.
        Returns:
            List[List[float]]: The training and validation losses.
        """
        self.model.train()

        sampler = torch.utils.data.DataLoader(self.train_set, batch_size, shuffle=True, generator=self.generator, collate_fn=self._collate_fn_pad)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3)
        # Reduce the learning rate if the loss does not decrease for some iterations.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        losses = [[],[]]
        sliding_training_loss = []

        epoch = 1
        iteration = 0
        end_time = time.time() + max_time
        last_state_dict = {}

        while True:
            for batch in tqdm.tqdm(sampler, total=len(sampler), desc=f"Epoch {epoch}"):
                try:
                    self.model.train()
                    
                    # The data was padded to make it loadable. Pack it to ignore padded data.
                    x = batch[:, :-1].to(self.device)
                    y = batch[:, 1:].to(self.device)
                    logits = self.model.forward(x)

                    # If the model returns a list, take the first element. Those are the logits.
                    if type(logits) is list or type(logits) is tuple:
                        logits = logits[0]

                    B, T, C = logits.shape
                    y = y.reshape(B*T)
                    logits = logits.reshape(B*T, C)
                    loss = torch.nn.functional.cross_entropy(logits, y, ignore_index=0) # Only compute loss on non-padded outputs.

                    # Compute the average loss over the last 100 iterations. This is used to determine when to decrease the learning rate.
                    sliding_training_loss.append(loss.detach().cpu().item())
                    if len(sliding_training_loss) > 100:
                        sliding_training_loss.pop(0)

                    optimizer.zero_grad()
                    loss.backward()

                    # Update the model weights.
                    optimizer.step()

                    iteration += 1

                    # If necessary, decrease the learning rate if the loss does not decrease for 100 iterations.
                    if iteration % 100 == 0:
                        avg_training_loss = sum(sliding_training_loss) / len(sliding_training_loss)
                        scheduler.step(avg_training_loss)

                    if iteration == max_iterations or (max_time > 0 and time.time() > end_time):
                        epoch = max_epochs
                        return
                    
                    # Show current performance every once in a while.
                    if iteration % progress_report == 0:
                        train_loss = self.test(test_set=self.train_set[:100], batch_size=batch_size)
                        valid_loss = self.test(test_set=self.valid_set[:100], batch_size=batch_size)
                        print(f"Training loss of current epoch: {train_loss}")
                        print(f"Validation loss of current epoch: {valid_loss}")
                except KeyboardInterrupt as e:
                    print("Saving model...")
                    self.save_model()
                    exit()

            losses[0].append(self.test(test_set=self.train_set[:100], batch_size=batch_size))
            losses[1].append(self.test(test_set=self.valid_set[:100], batch_size=batch_size))
            self.model.train()

            epoch += 1
            gain = (losses[1][-2] - losses[1][-1]) if len(losses[1]) > 1 else 1
            if gain < 0:
                # Validation error increased, load last state and abort.
                print(f"Validation loss increased. Resetting state to last epoch and aborting.")
                self.model.load_state_dict(last_state_dict)
                break
            if epoch > max_epochs:
                break

            last_state_dict = self.model.state_dict()

        return losses
    
    @torch.no_grad()
    def test(self, test_set: Any, batch_size: int = 16) -> float:
        """Tests the current model on the specified dataset.

        Args:
            valid_set (bool, optional): Whether to use the validation set instead of the test set. Defaults to False.
            batch_size (int, optional): The amount of data points to evaluate at once. Defaults to 4096 (fits in roughly 4GB of VRAM).

        Returns:
            float: The cross entropy loss.
        """
        self.model.eval()

        sampler = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True, generator=self.generator, collate_fn=self._collate_fn_pad)
        batches = len(sampler)

        avg_loss = 0
        for batch in sampler:
            # The data was padded to make it loadable. Pack it to ignore padded data.
            x = batch[:, :-1].to(self.device)
            y = batch[:, 1:].to(self.device)
            logits = self.model.forward(x)

            # If the model returns a list, take the first element. Those are the logits.
            if type(logits) is list or type(logits) is tuple:
                logits = logits[0]

            B, T, C = logits.shape
            y = y.reshape(B*T)
            logits = logits.reshape(B*T, C)
            loss = torch.nn.functional.cross_entropy(logits, y, ignore_index=0) # Only compute loss on non-padded outputs.

            avg_loss += loss.item() / batches

        return avg_loss

    @torch.no_grad()
    def generate_text(self, beginning: str = None, print_live: bool = True) -> str:
        """Generates text using the model. If a beginning is specified, the model will continue the text. Otherwise, it will generate a new text.
        The model will generate text until it reaches the end token. This may never happen if the model is not trained well enough.

        Args:
            beginning (str, optional): The beginning of the text. Defaults to None.
            print_live (bool, optional): Whether to print the generated text live. Defaults to True.
        Returns:
            str: The generated text.
        """
        self.model.eval()

        word_codes = []
        generated_text = "\n"
        # Start with 1 padding.
        context = [50259]

        # Fill context with recipe.
        if beginning:
            generated_text = beginning
            word_codes = encode_texts([beginning])[0]
            context += word_codes
        
        states = []

        # Fill the states if the model returns any.
        if self.context_length == 1:
            for value in context:
                logits = self.model.forward(torch.tensor([[value]]).to(self.device), *states)
                if type(logits) is list or type(logits) is tuple:
                    states = logits[1:]
                else:
                    break

        while True:
            # Model generated result for every index of context. We only need the last one.
            logits = self.model.forward(torch.tensor([context]).to(self.device), *states)

            # If the model returns a list, take the first element. Those are the logits. The rest are states for the next iteration.
            if type(logits) is list or type(logits) is tuple:
                states = logits[1:]
                logits = logits[0]

            # Take the last logit, which is the one for the last token.
            last_logit = logits[0, 0, :]

            # Sample from the logits. This is the next token.
            probs = torch.nn.functional.softmax(last_logit, dim=0)
            word_code = torch.multinomial(probs, num_samples=1).item()
            
            # Don't add padding.
            if word_code == 50259:
                continue
            
            context.append(word_code)
            if len(context) > self.context_length:
                context = context[1:]
                
            word_codes.append(word_code)

            # Make text look nicer.
            next_text = enc.decode([word_code]).replace("<|padding|>", "")\
                                                 .replace("<|next_step|>", "\n\n")\
                                                 .replace("<|ingredients_end|>", "\n\nInstructions:\n")\
                                                 .replace("<|endoftext|>", "\n\n")
            if print_live:
                print(next_text, end="")
            generated_text += next_text

            # End of text token reached. Stop.
            if word_code == 50256:
                break
        
        return generated_text

    def save_model(self, name: str = None):
        """Saves the current model to disk. The model can be loaded again using load_model.

        Args:
            name (str, optional): The name of the model. Defaults to None. If None, the name of the model class will be used.
        """
        if name is None:
            name = type(self.model).__name__

        torch.save(self.model.state_dict(), f"./Models/Instructions/{name}.pkl")

    def load_model(self, name: str = None):
        """Loads a model from disk. The model must have been saved using save_model.

        Args:
            name (str, optional): The name of the model. Defaults to None. If None, the name of the model class will be used.
        """
        if name is None:
            name = type(self.model).__name__

        state_dict = torch.load(f"./Models/Instructions/{name}.pkl")
        self.model.load_state_dict(state_dict)
