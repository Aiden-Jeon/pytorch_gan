import torch
import torch.nn as nn


class ActivatonLayer(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def forward(self, x):
        return self.activation(x) if self.activation is not None else x

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        batch_norm: bool = True,
        activation: str = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = ActivatonLayer(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.bn(y) if self.bn is not None else y
        y = self.activation(y)
        return y


class ConvTransposeLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        batch_norm: bool = True,
        activation: str = None,
    ):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = ActivatonLayer(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_trans(x)
        y = self.bn(y) if self.bn is not None else y
        y = self.activation(y)
        return y