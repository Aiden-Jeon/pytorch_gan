import torch.nn as nn
from layers import make_discriminator_layers


class Discriminator(nn.Module):
    def __init__(self, input_size: int, hidden_channel: int = 128):
        super().__init__()
        self.layers = nn.Sequential(*make_discriminator_layers(input_size, hidden_channel))

    def forward(self, x):
        return self.layers(x).view(-1, 1)
