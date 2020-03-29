import torch

from .base_gan import BaseGAN
from loss import bce_loss_with_logit
from modules import Generator
from modules import Discriminator


class DeepConvolutionGAN(BaseGAN):
    def __init__(
        self,
        generator_and_opt: [Generator, torch.optim] = None,
        discriminator_and_opt: [Discriminator, torch.optim] = None,
        input_size: int = None,
        hidden_channel: int = 128,
        latent_dim: int = 100,
        learning_rate: float = 1e-4,
        loss_fn: callable = bce_loss_with_logit
    ):
        super().__init__(generator_and_opt, discriminator_and_opt, input_size, hidden_channel, latent_dim, learning_rate)
        self.loss_fn = loss_fn

    def discriminator_loss(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        batch_size = x.size(0)
        device = x.device
        latent_dim = self.generator.latent_dim
        #
        # 1. get discriminator loss from real data
        #
        real_D_score = self.discriminator(x)
        real_target = torch.ones_like(real_D_score).to(device)
        real_D_loss = self.loss_fn(real_D_score, real_target)
        #
        # 2. get discriminator loss from fake data
        #
        z = torch.randn((batch_size, latent_dim)).to(device)
        fake_data = self.generator(z)

        fake_D_score = self.discriminator(fake_data)
        fake_target = torch.zeros_like(fake_D_score).to(device)
        fake_D_loss = self.loss_fn(fake_D_score, fake_target)

        return real_D_loss, fake_D_loss

    def generator_loss(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        latent_dim = self.generator.latent_dim

        z = torch.randn((batch_size, latent_dim)).to(device)
        fake_data = self.generator(z)

        fake_D_score = self.discriminator(fake_data)
        real_target = torch.ones_like(fake_D_score).to(device)

        G_loss = self.loss_fn(fake_D_score, real_target)

        return G_loss
