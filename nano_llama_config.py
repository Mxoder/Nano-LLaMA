import math
import torch
from typing import Optional
from dataclasses import dataclass

@dataclass
class NanoLlamaConfig:
    vocab_size: int = 256
    hidden_size: int = 96
    intermediate_size: int = int(math.ceil(hidden_size * 8/3 / 128)) * 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 1e4
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 4096
    dropout: float = 0.0
    mlp_bias: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
