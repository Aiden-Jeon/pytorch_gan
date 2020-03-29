from typing import Callable, Union, Any, Optional
import torch
import torch.nn.functional as F
from .base_gan import BaseGAN


class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(256, 1, 4)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output


class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(256, 128, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 100, 1)

        self.conv_mu = nn.Conv2d(128, 1, 1)
        self.conv_var = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        # Not used during training for celeba dataset.
        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var


class InfoGAN(BaseGAN):
    def __init__(self, generator, generator_opt, discriminator, discriminator_opt, latent_type):
        super().__init__(generator, generator_opt, discriminator, discriminator_opt)
        self.latent_type = latent_type
    
    def adversarial_loss_fn(self, score, target):
        prob = torch.sigmoid(score)
        return F.binary_cross_entropy(prob, target)
    
    def Q_loss_fn(self, score, target):
        prob = F.softmax(score)
        return F.cross_entropy(prob, target)
    
    def discriminator_loss(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        batch_size = x.size(0)
        device = x.device
        latent_dim = self.generator.latent_dim
        #
        # 1. get discriminator loss from real data
        #
        real_D_score = self.discriminator(x)
        real_D_score = 1 - real_D_score
        real_D_loss = self.loss_fn(real_D_score)
        #
        # 2. get discriminator loss from fake data
        #
        z = torch.randn((batch_size, latent_dim)).to(device)
        fake_data = self.generator(z)

        fake_D_score = self.discriminator(fake_data)
        fake_D_score = 1 + fake_D_score
        fake_D_loss = self.loss_fn(fake_D_score)

        return real_D_loss, fake_D_loss
    
    def generator_loss(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        latent_dim = self.generator.latent_dim

        z = torch.randn((batch_size, latent_dim)).to(device)
        fake_data = self.generator(z)

        fake_D_score = self.discriminator(fake_data)

        G_loss = fake_D_score.mul(-1).mean()

        return G_loss
