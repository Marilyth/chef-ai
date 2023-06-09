import torch
from torch import nn
from transformers import LEDModel, AutoModelForSeq2SeqLM
from Models.Instructions.ModuleBase import EncoderDecoderModuleBase
from Data import data


class FineTunedLED(EncoderDecoderModuleBase):
    def __init__(self):
        super().__init__()
        # Pretrained LED model without a head.
        self.model = LEDModel.from_pretrained("allenai/led-base-16384")

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
        global_attention_mask = torch.zeros(src.shape, dtype=torch.long, device=src.device)
        # Set global attention to the <s> token.
        global_attention_mask = global_attention_mask.masked_fill(src == data.tokenizer.cls_token_id, 1)
        
        outputs = self.model.forward(input_ids=src, decoder_input_ids=tgt, global_attention_mask=global_attention_mask).last_hidden_state

        # Apply the MLP on top of the LED output.
        outputs = self.mlp_in(outputs)
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        result = self.mlp_out(outputs)

        return result

class FineTunedLEDHug(EncoderDecoderModuleBase):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384")
        self.model.gradient_checkpointing = True
        self.model.config.num_beams = 4
        self.model.config.max_length = 512
        self.model.config.min_length = 100
        self.model.config.length_penalty = 2.0
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3

        self.source_length = 8192
        self.target_length = 512

        # Freeze all parameters except the head.
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.lm_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor = None, **model_inputs) -> torch.Tensor:
        # Get the last hidden state from the LED model.
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        # Set global attention to the <s> token.
        global_attention_mask = global_attention_mask.masked_fill(input_ids == data.tokenizer.cls_token_id, 1)

        outputs = self.model.forward(input_ids=input_ids, decoder_input_ids=decoder_input_ids, global_attention_mask=global_attention_mask)

        return outputs
    
    def generate(self, *args, **kwargs):
        input_ids = args[0]
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        # Set global attention to the <s> token.
        global_attention_mask = global_attention_mask.masked_fill(input_ids == data.tokenizer.cls_token_id, 1)
        kwargs["global_attention_mask"] = global_attention_mask
        
        return self.model.generate(*args, **kwargs)
    
class FineTunedLEDHugLarge(EncoderDecoderModuleBase):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384", gradient_checkpointing=True, use_cache=False)
        self.model.config.num_beams = 4
        self.model.config.max_length = 512
        self.model.config.min_length = 100
        self.model.config.length_penalty = 2.0
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3

        self.source_length = 8192
        self.target_length = 512

        # Freeze all parameters except the head.
        #for param in self.model.parameters():
        #    param.requires_grad = False

        #for param in self.model.lm_head.parameters():
        #    param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor = None, **model_inputs) -> torch.Tensor:
        # Get the last hidden state from the LED model.
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        # Set global attention to the <s> token.
        global_attention_mask = global_attention_mask.masked_fill(input_ids == data.tokenizer.cls_token_id, 1)

        outputs = self.model.forward(input_ids=input_ids, decoder_input_ids=decoder_input_ids, global_attention_mask=global_attention_mask)

        return outputs

    def generate(self, *args, **kwargs):
        input_ids = args[0]
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        # Set global attention to the <s> token.
        global_attention_mask = global_attention_mask.masked_fill(input_ids == data.tokenizer.cls_token_id, 1)
        kwargs["global_attention_mask"] = global_attention_mask
        
        return self.model.generate(*args, **kwargs)