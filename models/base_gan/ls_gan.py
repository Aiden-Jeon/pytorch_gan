import torch

from .dc_gan import DeepConvolutionGAN
from loss import mse_loss
from modules import Generator
from modules import Discriminator


class LeastSquaresGAN(DeepConvolutionGAN):
    def __init__(
        self,
        generator_and_opt: [Generator, torch.optim] = None,
        discriminator_and_opt: [Discriminator, torch.optim] = None,
        input_size: int = None,
        hidden_channel: int = 128,
        latent_dim: int = 100,
        learning_rate: float = 1e-4,
        loss_fn=mse_loss
    ):
        super().__init__(generator_and_opt, discriminator_and_opt, input_size, hidden_channel, latent_dim, learning_rate)
        self.loss_fn = loss_fn
