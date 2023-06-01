import torch
from torch import nn
from transformers import T5Model, LongT5ForConditionalGeneration
from Models.Instructions.ModuleBase import EncoderDecoderModuleBase


class FineTunedT5(EncoderDecoderModuleBase):
    def __init__(self):
        super().__init__()
        # Pretrained T5 model without a head.
        self.model = T5Model.from_pretrained("t5-small")
        self.source_length = self.model.config.n_positions
        self.target_length = self.model.config.n_positions

        # Add a simple MLP on top of the model to learn to convert the T5 output to the vocabulary.
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
        # Get the last hidden state from the T5 model.
        outputs = self.model.forward(input_ids=src, decoder_input_ids=tgt).last_hidden_state

        # Apply the MLP on top of the T5 output.
        outputs = self.mlp_in(outputs)
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        result = self.mlp_out(outputs)

        return result

class FineTunedLongT5(EncoderDecoderModuleBase):
    def __init__(self):
        super().__init__()
        # Pretrained T5 model without a head.
        self.model = LongT5ForConditionalGeneration.from_pretrained("whaleloops/longt5-tglobal-large-16384-pubmed-10k_steps")
        self.model.gradient_checkpointing_enable()
        self.model.config.num_beams = 4
        self.model.config.max_length = 512
        self.model.config.min_length = 100
        self.model.config.length_penalty = 2.0
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.source_length = self.model.config.n_positions
        self.target_length = self.model.config.n_positions

        # Freeze all parameters except the head.
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        self.save_hyperparameters()
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Get the last hidden state from the T5 model.
        outputs = self.model.forward(input_ids=src, decoder_input_ids=tgt)

        return outputs
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)