from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
import pytorch_lightning as lightning
from abc import ABC
import optuna


class DecoderOnlyBase(lightning.LightningModule, ABC):
    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, log_name: str) -> torch.Tensor:
        """Performs a training or validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.
            mode (str): The mode of the step.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        # Get the input and target.
        input = batch[:, :-1]
        target = batch[:, 1:]

        # Perform a forward pass.
        output = self(input)

        # If the model returns a list, take the first element. Those are the logits.
        if type(output) is list or type(output) is tuple:
            output = output[0]
        
        B, T, C = output.shape
        target = target.reshape(B*T)
        output = output.reshape(B*T, C)

        # Calculate the loss.
        loss = nn.CrossEntropyLoss()(output, target, ignore_index=tokenizer.pad_token_id)

        # Log the loss.
        self.log(log_name, loss, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        return self.step(batch, batch_idx, "train_loss")
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        return self.step(batch, batch_idx, "val_loss")
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        return self.step(batch, batch_idx, "test_loss")
    
    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """Configures the optimizer and scheduler.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]: The optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        monitor = "val_loss"
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def get_optuna_parameters(self, trial: optuna.Trial) -> List[Any]:
        """Gets the parameters to optimize using optuna.

        Args:
            trial (optuna.Trial): The trial.

        Returns:
            List[Any]: The parameters for the next objective step.
        """
        return []

    def optuna_optimize(self, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, n_trials: int = 10, n_jobs: int = 1, time_per_trial: str = "00:00:05:00") -> Tuple[Dict[str, Any], float]:
        """Optimizes the model using optuna.

        Args:
            model (lightning.LightningModule): The model to optimize.
            train_dataset (torch.utils.data.Dataset): The training dataset.
            val_dataset (torch.utils.data.Dataset): The validation dataset.
            n_trials (int, optional): The amount of trials to run. Defaults to 10.
            n_jobs (int, optional): The amount of jobs to run in parallel. Defaults to 1.

        Returns:
            Tuple[Dict[str, Any], float]: The best parameters and the best score.
        """
        def objective(trial: optuna.Trial) -> float:
            """The objective function to optimize.

            Args:
                trial (optuna.Trial): The trial.

            Returns:
                float: The score to optimize.
            """
            # Get the parameters to optimize.
            parameters = self.get_optuna_parameters(trial)
            trial_model = type(self)(*parameters)
            
            # Create the trainer.
            trainer = lightning.Trainer(
                logger=True,
                deterministic=True, max_time=time_per_trial,
                enable_checkpointing=False,
                val_check_interval=500,
                callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
            )

            # Train the model.
            trainer.fit(trial_model, train_dataset, val_dataset)

            # Return the best score.
            return trainer.callback_metrics["val_loss"]

        # Create the study.
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        # Optimize the study.
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=(RuntimeError))

        # Return the best parameters and the best score.
        return study.best_params, study.best_value

    @torch.no_grad()
    def generate(self, beginning: str = None, print_live: bool = True, temperature: float = 1.0, top_k: int = -1, top_p: float = 1.0) -> str:
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
        self.eval()

        word_codes = []
        generated_text = "\n"
        # Start with 1 padding.
        context = [tokenizer.pad_token_id]

        # Fill context with recipe.
        if beginning:
            generated_text = beginning
            word_codes = encode_texts([beginning])[0]
            context += word_codes
        
        states = []

        if not hasattr(self, "context_length"):
            self.context_length = 1

        # Fill the states if the model returns any.
        if self.context_length == 1:
            for value in context:
                logits = self.forward(torch.tensor([[value]]).to(self.device), *states)
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
                logits = self.forward(torch.tensor([context]).to(self.device), *states)

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
                if word_code == tokenizer.pad_token_id:
                    continue
                
                context.append(word_code)
                if len(context) > self.context_length:
                    context = context[1:]
                    
                word_codes.append(word_code)

                # Make text look nicer.
                next_text = tokenizer.decode([word_code]).replace("<|padding|>", "")\
                                                    .replace("<|ingredients_end|>", "\n\nInstructions:\n")\
                                                    .replace("<|endoftext|>", "\n\n")\
                                                    .replace("<|next_step|>", "\n")\
                                                    .replace("\r", "\n")
                
                if print_live:
                    print(next_text, end="")
                generated_text += next_text

                # End of text token reached. Stop.
                if word_code == tokenizer.eos_token_id:
                    break
            except KeyboardInterrupt as e:
                # Stop on keyboard interrupt.
                print()
                break
        
        return generated_text
    
class EncoderDecoderModuleBase(lightning.LightningModule):
    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, log_name: str) -> torch.Tensor:
        """Performs a training or validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.
            mode (str): The mode of the step.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        # Get the input and target.
        encoder_input = batch[0]
        input = batch[1][:, :-1]
        target = batch[1][:, 1:]

        test_encoder = tokenizer.decode(encoder_input[0])
        test_input = tokenizer.decode(input[0])
        test_target = tokenizer.decode(target[0])

        # Perform a forward pass.
        output = self(encoder_input, input)

        # If the model returns a list, take the first element. Those are the logits.
        if type(output) is list or type(output) is tuple:
            output = output[0]
        
        B, T, C = output.shape
        target = target.reshape(B*T)
        output = output.reshape(B*T, C)

        # Calculate the loss.
        loss = nn.functional.cross_entropy(output, target, ignore_index=tokenizer.pad_token_id)

        # Log the loss.
        self.log(log_name, loss, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        return self.step(batch, batch_idx, "train_loss")
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        return self.step(batch, batch_idx, "val_loss")
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        return self.step(batch, batch_idx, "test_loss")
    
    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """Configures the optimizer and scheduler.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]: The optimizer and scheduler.
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1, verbose=True)
        self.monitor = "val_loss"
        return {"optimizer": self.optimizer, "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": self.monitor,
                "interval": "step",
                "frequency": 500
            }}
    
    def optuna_optimize(self, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, n_trials: int = 10, n_jobs: int = 1, time_per_trial: str = "00:00:07:30") -> Tuple[Dict[str, Any], float]:
        """Optimizes the model using optuna.

        Args:
            model (lightning.LightningModule): The model to optimize.
            train_dataset (torch.utils.data.Dataset): The training dataset.
            val_dataset (torch.utils.data.Dataset): The validation dataset.
            n_trials (int, optional): The amount of trials to run. Defaults to 10.
            n_jobs (int, optional): The amount of jobs to run in parallel. Defaults to 1.

        Returns:
            Tuple[Dict[str, Any], float]: The best parameters and the best score.
        """
        def objective(trial: optuna.Trial) -> float:
            """The objective function to optimize.

            Args:
                trial (optuna.Trial): The trial.

            Returns:
                float: The score to optimize.
            """
            # Get the parameters to optimize.
            parameters = self.get_optuna_parameters(trial)
            trial_model = type(self)(*parameters)
            
            # Create the trainer.
            trainer = lightning.Trainer(
                logger=True,
                deterministic=True, max_time=time_per_trial,
                enable_checkpointing=False,
                val_check_interval=500,
                callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
            )

            # Train the model.
            trainer.fit(trial_model, train_dataset, val_dataset)

            # Return the best score.
            return trainer.callback_metrics["val_loss"]

        # Create the study.
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        # Optimize the study.
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=(RuntimeError))

        # Return the best parameters and the best score.
        return study.best_params, study.best_value
    
    @torch.no_grad()
    def generate(self, encoder_text: str = None, temperature: float = 1.0, top_k: int = -1, top_p: float = 1.0, print_live: bool = False, truncate: bool = True) -> str:
        """Generates text using the model. If a beginning is specified, the model will continue the text. Otherwise, it will generate a new text.
        The model will generate text until it reaches the end token. This may never happen if the model is not trained well enough.

        Args:
            encoder_text (str, optional): The text to use as the encoder input. Defaults to None.
            temperature (float, optional): The temperature to use when generating text. Defaults to 1.0. Higher values will result in more random text.
            top_k (int, optional): The amount of top k words to use when generating text. Defaults to -1. This will use all words.
            top_p (float, optional): The probability to use when generating text. Defaults to 1.0. Higher values will result in more random text.
            compression (int, optional): The amount of compression to use when generating text, i.e. the amount of times to loop over the generated output. Defaults to 0.
        Returns:
            str: The generated text.
        """
        self.eval()
        self.cuda()
        
        word_codes = []
        printed_text = ""
        chunk = encoder_text
        generated_text = "\n"
        # Start with start token.
        context = [tokenizer.pad_token_id]

        # Fill context with recipe.
        #encoder_text = encode_texts([encoder_text])
        # Shape into source length.
        if truncate:
            encoder_tokens = tokenizer(chunk, max_length=self.source_length - 1, truncation=True, padding=True).data['input_ids']
        else:
            encoder_tokens = tokenizer(chunk).data['input_ids']
        encoder_tokens = torch.tensor(add_start_tokens([encoder_tokens])).to(self.device)
        
        states = []

        while True:
            try:
                # Model generated result for every index of context. We only need the last one.
                logits = self.forward(encoder_tokens, torch.tensor([context]).to(self.device), *states)

                # If the model returns a list, take the first element. Those are the logits. The rest are states for the next iteration.
                if type(logits) is list or type(logits) is tuple:
                    states = logits[1:]
                    logits = logits[0]

                # Take the last logit, which is the one for the last token.
                last_logit = logits[-1, -1, :]

                # Sample from the logits. This is the next token. The higher the temperature, the more random the text will be.
                probs = torch.nn.functional.softmax(last_logit / temperature, dim=0)

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
                if word_code == tokenizer.pad_token_id:
                    continue
                
                context.append(word_code)
                if len(context) > self.target_length:
                    context = context[1:]
                    
                word_codes.append(word_code)

                # Print the generated text so far. Only print the new part.
                if print_live:
                    current_text = tokenizer.decode(word_codes[-10:]).replace("</s>", "\n")
                    if printed_text == "":
                        print(current_text, end="")
                    else:
                        # Find the overlap between the current text and the printed text.
                        overlap = 0
                        for i in range(1, len(printed_text) + 1):
                            if current_text.startswith(printed_text[-i:]):
                                overlap = i

                        # Print the new part.
                        print(current_text[overlap:].rstrip(",?! \"'.\n"), end="")
                        if overlap == 0:
                            print()

                    printed_text = current_text.rstrip(",?! \"'.\n")

                # End of text token reached. Stop.
                if word_code == tokenizer.eos_token_id:
                    break
            except KeyboardInterrupt as e:
                # Stop on keyboard interrupt.
                print()
                break
    
        generated_text = tokenizer.decode(word_codes).replace("</s>", "\n")
        encoder_text = generated_text

        return generated_text

    @torch.no_grad()
    def generate_chunked(self, encoder_text: str = None, temperature: float = 1.0, top_k: int = -1, top_p: float = 1.0, compression: int = 0, print_live: bool = False) -> str:
        """Generates text using the model. If a beginning is specified, the model will continue the text. Otherwise, it will generate a new text.
        The model will generate text until it reaches the end token. This may never happen if the model is not trained well enough.

        Args:
            encoder_text (str, optional): The text to use as the encoder input. Defaults to None.
            temperature (float, optional): The temperature to use when generating text. Defaults to 1.0. Higher values will result in more random text.
            top_k (int, optional): The amount of top k words to use when generating text. Defaults to -1. This will use all words.
            top_p (float, optional): The probability to use when generating text. Defaults to 1.0. Higher values will result in more random text.
            compression (int, optional): The amount of compression to use when generating text, i.e. the amount of times to loop over the generated output. Defaults to 0.
        Returns:
            str: The generated text.
        """
        self.eval()
        self.cuda()
        
        for loop in tqdm.tqdm(range(compression + 1), desc="Iteration", disable=print_live):
            word_codes = []
            printed_text = ""
            for chunk in tqdm.tqdm(chunk_text(encoder_text), desc="Generating", disable=print_live):
                generated_text = "\n"
                # Start with start token.
                context = [tokenizer.pad_token_id]

                # Fill context with recipe.
                #encoder_text = encode_texts([encoder_text])
                # Shape into source length.
                encoder_tokens = tokenizer(chunk, max_length=self.source_length - 1, truncation=True, padding=True).data['input_ids']
                encoder_tokens = torch.tensor(add_start_tokens([encoder_tokens])).to(self.device)
                
                states = []

                while True:
                    try:
                        # Model generated result for every index of context. We only need the last one.
                        logits = self.forward(encoder_tokens, torch.tensor([context]).to(self.device), *states)

                        # If the model returns a list, take the first element. Those are the logits. The rest are states for the next iteration.
                        if type(logits) is list or type(logits) is tuple:
                            states = logits[1:]
                            logits = logits[0]

                        # Take the last logit, which is the one for the last token.
                        last_logit = logits[-1, -1, :]

                        # Sample from the logits. This is the next token. The higher the temperature, the more random the text will be.
                        probs = torch.nn.functional.softmax(last_logit / temperature, dim=0)

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
                        if word_code == tokenizer.pad_token_id:
                            continue
                        
                        context.append(word_code)
                        if len(context) > self.target_length:
                            context = context[1:]
                            
                        word_codes.append(word_code)

                        # Print the generated text so far. Only print the new part.
                        if print_live:
                            current_text = tokenizer.decode(word_codes[-10:]).replace("</s>", "\n")
                            if printed_text == "":
                                print(current_text, end="")
                            else:
                                # Find the overlap between the current text and the printed text.
                                overlap = 0
                                for i in range(1, len(printed_text) + 1):
                                    if current_text.startswith(printed_text[-i:]):
                                        overlap = i

                                # Print the new part.
                                print(current_text[overlap:].rstrip(",?! \"'.\n"), end="")
                                if overlap == 0:
                                    print()

                            printed_text = current_text.rstrip(",?! \"'.\n")

                        # End of text token reached. Stop.
                        if word_code == tokenizer.eos_token_id:
                            break
                    except KeyboardInterrupt as e:
                        # Stop on keyboard interrupt.
                        print()
                        break
            
            generated_text = tokenizer.decode(word_codes).replace("</s>", "\n")
            encoder_text = generated_text

            if print_live:
                print("\n\n")
        return generated_text


from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor
        self.step = 0

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            print(message)
            return

        self._trial.report(current_score, step=self.step)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(self.step)
            raise optuna.TrialPruned(message)
        self.step += 1