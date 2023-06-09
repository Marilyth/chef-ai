import torch
from torch import nn
from transformers import BartModel
from Models.Instructions.ModuleBase import EncoderDecoderModuleBase


class FineTunedBart(EncoderDecoderModuleBase):
    def __init__(self):
        super().__init__()
        # Pretrained BART model without a head.
        self.model = BartModel.from_pretrained("facebook/bart-base")

        # Add a simple MLP on top of the model to learn to convert the BART output to the vocabulary.
        # This can also be a single layer, but the performance is better with a more complex structure.
        self.mlp_in = nn.Linear(self.model.config.d_model, self.model.config.d_model, bias=False)
        self.norm = nn.LayerNorm(self.model.config.d_model)
        self.activation = nn.ReLU()
        self.mlp_out = nn.Linear(self.model.config.d_model, self.model.config.vocab_size, bias=False)

        # Freeze model parameters if desired. Otherwise the complete model will be fine-tuned.
        # If all layers are frozen, training is much faster, but the performance might suffer.
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        self.save_hyperparameters()
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Get the last hidden state from the BART model.
        outputs = self.model.forward(input_ids=src, decoder_input_ids=tgt).last_hidden_state

        # Apply the MLP on top of the BART output.
        outputs = self.mlp_in(outputs)
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        result = self.mlp_out(outputs)

        return result
    