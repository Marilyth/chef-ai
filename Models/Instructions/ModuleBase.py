from typing import *
import torch
import torch.nn as nn
import torch.utils.data
from Data.data import *
import tqdm
import time
import lightning
from abc import ABC


class ModuleBase(lightning.LightningModule, ABC):
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
        loss = nn.CrossEntropyLoss()(output, target)

        # Log the loss.
        self.log(log_name, loss)

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
    
    def configure_optimizers(self):
        """Configures the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)
