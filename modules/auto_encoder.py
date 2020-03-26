import torch
import torch.nn as nn
from layers import make_encoder_layers, make_decoder_layers


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_channel: int = 128, latent_dim: int = 100):
        super().__init__()
        self.encoder = nn.Sequential(*make_encoder_layers(input_size, hidden_channel, latent_dim))
        self.decoder = nn.Sequential(*make_decoder_layers(input_size, hidden_channel, latent_dim))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z