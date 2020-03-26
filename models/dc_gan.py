from typing import Callable, Union, Any, Optional
import torch
import torch.nn.functional as F
from .base_gan import BaseGAN


class DeepConvolutionGAN(BaseGAN):
    def __init__(self, generator, generator_opt, discriminator, discriminator_opt):
        super().__init__(generator, generator_opt, discriminator, discriminator_opt)

    def loss_fn(self, score, target):
        prob = torch.sigmoid(score)
        return F.binary_cross_entropy(prob, target)
        
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


class LeastSquaresGAN(DeepConvolutionGAN):
    def __init__(self, generator, generator_opt, discriminator, discriminator_opt):
        super().__init__(generator, generator_opt, discriminator, discriminator_opt)

    def loss_fn(self, score, target):
        return torch.nn.BCELoss()(score, target)