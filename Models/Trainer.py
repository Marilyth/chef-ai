from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
import lightning

torch.manual_seed(42)
class Trainer:
    def __init__(self, model: nn.Module, context_length: int = 1, load_checkpoint: bool = True):
        """Initializes a trainer for the specified model.

        Args:
            model (nn.Module): The model to train.
            context_length (int): The context length the predict the next word with during generation. For RNN models this is 1.
        """
        lightning.Trainer()

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
        texts = encode_texts(get_texts(file_name, size, torch.randint(0, 100, (1,)).item()))

        # This data is of variable length. It needs to be packed before forward pass.
        data = [torch.tensor(datapoint) for datapoint in create_sliding(texts, context_length + 1, 50259)]

        train_length = int(0.8 * len(data))
        valid_length = int(0.1 * len(data))
        test_length = len(data) - train_length - valid_length

        train_set, valid_set, test_set = torch.utils.data.random_split(data, [train_length, valid_length, test_length])
        self.train_set, self.valid_set, self.test_set = [train_set.dataset[i] for i in train_set.indices],\
                                                        [valid_set.dataset[i] for i in valid_set.indices],\
                                                        [test_set.dataset[i] for i in test_set.indices]

    def train(self, max_epochs: int = 20, max_time: int = -1, max_iterations: int = -1, batch_size: int = 8, progress_report: int = 1000, use_half: bool = False) -> List[List[float]]:
        pass
    
    @torch.no_grad()
    def test(self, test_set: Any, batch_size: int = 16, show_progress: bool = False) -> float:
        """Tests the current model on the specified dataset.

        Args:
            valid_set (bool, optional): Whether to use the validation set instead of the test set. Defaults to False.
            batch_size (int, optional): The amount of data points to evaluate at once. Defaults to 4096 (fits in roughly 4GB of VRAM).
            show_progress (bool, optional): Whether to show a progress bar. Defaults to False.

        Returns:
            float: The cross entropy loss.
        """
        self.model.eval()

        sampler = torch.utils.data.DataLoader(test_set, batch_size, collate_fn=self._collate_fn_pad)
        batches = len(sampler)

        avg_loss = 0
        for batch in tqdm.tqdm(sampler, total=len(sampler), desc=f"Testing") if show_progress else sampler:
            # The data was padded to make it loadable. Pack it to ignore padded data.
            x = batch[:, :-1].to(self.device)
            y = batch[:, 1:].to(self.device)
            logits = self.model.forward(x)

            # If the model returns a list, take the first element. Those are the logits.
            if type(logits) is list or type(logits) is tuple:
                logits = logits[0]

            B, T, C = logits.shape
            y = y.view(B*T)
            logits = logits.view(B*T, C)
            loss = torch.nn.functional.cross_entropy(logits, y, ignore_index=0) # Only compute loss on non-padded outputs.

            avg_loss += loss.item() / batches

        return avg_loss

    @torch.no_grad()
    def generate_text(self, beginning: str = None, print_live: bool = True, temperature: float = 1.0, top_k: int = -1, top_p: float = 1.0) -> str:
        """Generates text using the model. If a beginning is specified, the model will continue the text. Otherwise, it will generate a new text.
        The model will generate text until it reaches the end token. This may never happen if the model is not trained well enough.

        Args:
            beginning (str, optional): The beginning of the text. Defaults to None.
            print_live (bool, optional): Whether to print the generated text live. Defaults to True.
            temperature (float, optional): The temperature to use when generating text. Defaults to 1.0. Higher values will result in more random text.
            top_k (int, optional): The amount of top k words to use when generating text. Defaults to -1. This will use all words.
            top_p (float, optional): The probability to use when generating text. Defaults to 1.0. Higher values will result in more random text.
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
        
        # Limit context to the specified length.
        if len(context) > self.context_length:
            context = context[-self.context_length:]

        while True:
            try:
                # Model generated result for every index of context. We only need the last one.
                logits = self.model.forward(torch.tensor([context]).to(self.device), *states)

                # If the model returns a list, take the first element. Those are the logits. The rest are states for the next iteration.
                if type(logits) is list or type(logits) is tuple:
                    states = logits[1:]
                    logits = logits[0]

                # Take the last logit, which is the one for the last token.
                last_logit = logits[-1, -1, :]

                # Sample from the logits. This is the next token. The higher the temperature, the more random the text will be.
                probs = torch.nn.functional.softmax(last_logit / temperature, dim=0)

                # If the last token was a space or <|next_step|>, we don't want to generate one again.
                if context[-1] == 220:# or context[-1] == 50258:
                    probs[220] = 0
                    #probs[50258] = 0
                    probs = probs / torch.sum(probs)

                # Take the top k words. This will set all other words to 0.
                if top_k > 0:
                    top_k = min(top_k, probs.size(-1))
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
                    probs = probs / torch.sum(probs)

                # Take the top p words.
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    cumulative_mask = cumulative_probs > top_p

                    # Shift the mask to the right to keep the first token above the threshold.
                    cumulative_mask[..., 1:] = cumulative_mask[..., :-1].clone()
                    # Set the first token to False, because it should always be included.
                    cumulative_mask[0] = False

                    sorted_probs = sorted_probs.masked_fill(cumulative_mask, 0.0)

                    # If all probabilities are 0, set the first one to 1.0.
                    if sorted_probs[0] == 0.0:
                        sorted_probs[0] = 1.0

                    probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)
                    probs = probs / torch.sum(probs)

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
                                                    .replace("<|ingredients_end|>", "\n\nInstructions:\n")\
                                                    .replace("<|endoftext|>", "\n\n")\
                                                    .replace("<|next_step|>", "\n")\
                                                    .replace("\r", "\n")
                
                if print_live:
                    print(next_text, end="")
                generated_text += next_text

                # End of text token reached. Stop.
                if word_code == 50256:
                    break
            except KeyboardInterrupt as e:
                # Stop on keyboard interrupt.
                print()
                break
        
        return generated_text

    def save_checkpoint(self, name: str = None, assert_equal_loss: bool = True, save_training_state: bool = True):
        """Saves the current model to disk. The model can be loaded again using load_model.

        Args:
            name (str, optional): The name of the model. Defaults to None. If None, the name of the model class will be used.
            assert_equal_loss (bool, optional): Whether to assert that the loss is equal after loading the model. Defaults to True.
        """
        if name is None:
            name = type(self.model).__name__

        # Move model to cpu so that state_dict is not tied to a gpu.
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            "iteration": self.iteration
        } if save_training_state else {
            'model_state_dict': self.model.state_dict(),
        }

        torch.save(checkpoint, f"./Models/{name}.pkl")
        del checkpoint

        print("Saved.")

        if assert_equal_loss:
            print("Asserting equal loss...")
            loss_before = self.test(self.test_set[:100], show_progress=True, batch_size=16)
            self.load_checkpoint(name)
            loss_after = self.test(self.test_set[:100], show_progress=True, batch_size=16)

            print(f"{loss_before=}, {loss_after=}")

            assert round(loss_before, 2) == round(loss_after, 2), "Loss before and after saving are not equal."

    def load_checkpoint(self, name: str = None):
        """Loads a model from disk if it exists.

        Args:
            name (str, optional): The name of the model. Defaults to None. If None, the name of the model class will be used.
        """
        if name is None:
            name = type(self.model).__name__

        if os.path.exists(f"./Models/{name}.pkl"):
            checkpoint = torch.load(f"./Models/{name}.pkl")
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load training state if it exists.
            #if "optimizer_state_dict" in checkpoint:
            #    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #if "scheduler_state_dict" in checkpoint:
            #    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if "iteration" in checkpoint:
                self.iteration = checkpoint["iteration"]

            del checkpoint
            torch.cuda.empty_cache()
