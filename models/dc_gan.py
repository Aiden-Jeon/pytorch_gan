from typing import Callable, Union, Any, Optional

import torch
import torch.nn as nn

from models import AbstractGAN


class DCGAN(AbstractGAN):
    def __init__(self, generator, generator_opt, discriminator, discriminator_opt):
        super().__init__()
        self.generator = generator
        self.generator_opt = generator_opt
        self.discriminator = discriminator
        self.discriminator_opt = discriminator_opt
        self.loss_fn = nn.BCELoss()

    def get_discriminator_loss(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        #
        # 1. get discriminator loss from real data
        #
        real_D_score = self.discriminator(x)
        real_target = torch.ones_like(real_D_score).to(device)
        real_D_loss = self.loss_fn(real_D_score, real_target)
        #
        # 2. get discriminator loss from fake data
        #
        z = torch.randn((batch_size, self.generator.latent_dim)).to(device)
        fake_data = self.generator(z)
        
        fake_D_score = self.discriminator(fake_data)
        fake_target = torch.zeros_like(fake_D_score).to(device)
        fake_D_loss = self.loss_fn(fake_D_score, fake_target)

        return real_D_loss, fake_D_loss

    def get_generator_loss(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        z = torch.randn((batch_size, self.generator.latent_dim)).to(device)
        fake_data = self.generator(z)
        
        fake_D_score = self.discriminator(fake_data)
        real_target = torch.ones_like(fake_D_score).to(device)
        
        G_loss = self.loss_fn(fake_D_score, real_target)

        return G_loss

    def fit(self, batch: Optional[Union[tuple, list]]) -> dict:
        self.generator.train()
        self.discriminator.train()
        self.generator_opt.zero_grad()
        self.discriminator_opt.zero_grad()

        device = next(self.generator.parameters()).device
        x, y = batch
        x = x.to(device)
        #
        # 1. get discriminator loss and update discriminator
        #
        real_D_loss, fake_D_loss = self.get_discriminator_loss(x)
        D_loss = real_D_loss + fake_D_loss
        D_loss.backward()
        self.discriminator_opt.step()
        #
        # 2. get generator loss and update generator
        #
        G_loss = self.get_generator_loss(x)
        G_loss.backward()
        self.generator_opt.step()

        return {
            "D_loss": float(D_loss),
            "G_loss": float(G_loss),
        }