from typing import Union
from typing import Optional

import torch
from ignite.engine import Engine

from modules import Generator
from modules import Discriminator


class BaseGAN(torch.nn.Module):
    def __init__(
        self,
        generator_and_opt: [Generator, torch.optim] = None,
        discriminator_and_opt: [Discriminator, torch.optim] = None,
        input_size: int = None,
        hidden_channel: int = 128,
        latent_dim: int = 100,
        learning_rate: float = 1e-4,
    ):
        self.generator = None
        self.discriminator = None
        super().__init__()
        # Generator
        if generator_and_opt is None:
            assert input_size is None, "generator_and_opt or input_size should be given."
            if self.generator is not None:
                self.generator = Generator(input_size=input_size, latent_dim=latent_dim, hidden_channel=hidden_channel)
                self.generator_opt = torch.optim.Adam(self.generator.parameters(), learning_rate)
        else:
            self.generator, self.generator_opt = generator_and_opt
        # Discriminator
        if discriminator_and_opt is None:
            assert input_size is None, "discriminator_and_opt or input_size should be given."
            if self.discriminator is not None:
                self.discriminator = Discriminator(input_size=input_size, hidden_channel=hidden_channel)
                self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), learning_rate)
        else:
            self.discriminator, self.discriminator_opt = discriminator_and_opt

    def discriminator_loss(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return None, None

    def generator_loss(self, x: torch.Tensor) -> torch.Tensor:
        return None

    def fit_batch(self, engine: Engine, batch: Optional[Union[tuple, list]]) -> dict:
        return self.fit(batch)

    def fit(self, batch: Optional[Union[tuple, list]]) -> dict:
        self.generator.train()
        self.discriminator.train()
        self.generator_opt.zero_grad()
        self.discriminator_opt.zero_grad()

        device = next(self.generator.parameters()).device
        x, _ = batch
        x = x.to(device)
        #
        # 1. get discriminator loss and update discriminator
        #
        real_D_loss, fake_D_loss = self.discriminator_loss(x)
        D_loss = real_D_loss + fake_D_loss
        D_loss.backward()
        self.discriminator_opt.step()
        #
        # 2. get generator loss and update generator
        #
        G_loss = self.generator_loss(x)
        G_loss.backward()
        self.generator_opt.step()
        return {
            "D_loss": float(D_loss),
            "G_loss": float(G_loss),
        }