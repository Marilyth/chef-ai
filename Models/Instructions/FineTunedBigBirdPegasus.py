import torch
from torch import nn
from transformers import BigBirdPegasusModel
from Models.Instructions.ModuleBase import EncoderDecoderModuleBase


class FineTunedBBP(EncoderDecoderModuleBase):
    def __init__(self):
        super().__init__()
        # Pretrained LED model without a head.
        self.model = BigBirdPegasusModel.from_pretrained("google/bigbird-pegasus-large-pubmed")

        # Add a simple MLP on top of the model to learn to convert the LED output to the vocabulary.
        # This can also be a single layer, but the performance is better with a more complex structure.
        self.mlp_in = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.model.config.hidden_size)
        self.activation = nn.ReLU()
        self.mlp_out = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)

        # Freeze model parameters if desired. Otherwise the complete model will be fine-tuned.
        # If all layers are frozen, training is much faster, but the performance might suffer.
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        self.save_hyperparameters()
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Get the last hidden state from the LED model.
        #global_attention_mask = torch.ones(src.shape, dtype=torch.long, device=src.device)
        # Set mask to 2 for all tokens that are equal to 4, 2 or 0.
        #global_attention_mask = global_attention_mask.masked_fill(src == 0, 2)
        #global_attention_mask[:, [0,-1]] = 2
        
        outputs = self.model.forward(input_ids=src, decoder_input_ids=tgt).last_hidden_state

        # Apply the MLP on top of the LED output.
        outputs = self.mlp_in(outputs)
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        result = self.mlp_out(outputs)

        return result
    