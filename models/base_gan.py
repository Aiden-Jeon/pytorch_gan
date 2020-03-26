from typing import Callable, Union, Any, Optional
import torch
from ignite.engine import Engine


class BaseGAN(torch.nn.Module):
    def __init__(self, generator, generator_opt, discriminator, discriminator_opt):
        super().__init__()
        self.generator = generator
        self.generator_opt = generator_opt
        self.discriminator = discriminator
        self.discriminator_opt = discriminator_opt
    
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
        x, y = batch
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