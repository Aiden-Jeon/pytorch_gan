import torch.nn as nn
from layers import make_generator_layers


class Generator(nn.Module):
    def __init__(self, input_size: int, latent_dim: int = 100, hidden_channel: int = 128, last_act: str = "sigmoid"):
        super().__init__()
        self.layers = nn.Sequential(*make_generator_layers(input_size, latent_dim, hidden_channel, last_act))
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.layers(x.view(-1, self.latent_dim, 1, 1))
