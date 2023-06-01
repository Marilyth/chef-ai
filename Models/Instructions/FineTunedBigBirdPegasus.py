import torch
from torch import nn
from transformers import BigBirdPegasusForConditionalGeneration
from Models.Instructions.ModuleBase import EncoderDecoderModuleBase


class FineTunedBBP(EncoderDecoderModuleBase):
    def __init__(self):
        super().__init__()
        # Pretrained LED model without a head.
        self.model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", gradient_checkpointing=True)

        # Freeze model parameters if desired. Otherwise the complete model will be fine-tuned.
        # If all layers are frozen, training is much faster, but the performance might suffer.
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        self.save_hyperparameters()
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        outputs = self.model.forward(input_ids=src, decoder_input_ids=tgt)

        return outputs
    