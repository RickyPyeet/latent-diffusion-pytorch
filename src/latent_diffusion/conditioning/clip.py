import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel

class FrozenCLIP(nn.Module):
    def __init__(self, clip_name: str):
    """
    Instantiate a pretrained text encoder from clip_name, freezes it and encodes prompts
    args:
      clip_name = name of pretrained clip encoder - "openai/clip-vit-large-patch14"
    """
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.encoder = CLIPTextModel.from_pretrained(clip_name)

        for param in self.encoder.parameters():
            if param.requires_grad:
                param.requires_grad = False

        self.encoder.eval()

    @torch.no_grad()
    def encode(self, prompts: str | list[str]):
        """
        Takes a prompt or a list of prompts, turning that into tokens and then into embeddings.
        The prompts are padded and truncated equal to the max length handled by the tokenizer (e.g. 77)
        out:
            embeddigs (tensor) = embedded sequence of tokens
        """
        # Extract device
        device = next(self.encoder.parameters()).device
        # Create tokens
        tokens = self.tokenizer(prompts,
                                padding = 'max_length',
                                max_length = self.tokenizer.model_max_length,
                                truncation = True,
                                return_tensors = 'pt')
        # move tokens to device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        embeddings = self.encoder(**tokens).last_hidden_state
        
        return embeddings
