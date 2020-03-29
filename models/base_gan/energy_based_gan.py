import torch
import torch.nn.functional as F

from .base_gan import BaseGAN
from loss import mse_loss
from modules import Generator
from modules import AutoEncoder


class EnergyBasedGAN(BaseGAN):
    """
    margin
      - MNIST : 10
      - LSUN : 80
      - CelebA : 20
      - ImageNet
        - 128 x 128 : 40
        - 256x 256 : 80
    """

    def __init__(
        self,
        generator_and_opt: [Generator, torch.optim] = None,
        discriminator_and_opt: [AutoEncoder, torch.optim] = None,
        input_size: int = None,
        hidden_channel: int = 128,
        latent_dim: int = 100,
        learning_rate: float = 1e-4,
        margin: float = 20.0,
        lambda_pt: float = 0.1,
        loss_fn: callable = mse_loss,
    ):
        if discriminator_and_opt is None:
            assert input_size is None, "discriminator_and_opt or input_size should be given."
            if self.discriminator is not None:
                self.discriminator = AutoEncoder(input_size=input_size, hidden_channel=hidden_channel, latent_dim=latent_dim)
                self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), learning_rate)
        else:
            self.discriminator, self.discriminator_opt = discriminator_and_opt

        super().__init__(generator_and_opt, discriminator_and_opt, input_size, hidden_channel, latent_dim, learning_rate)
            
        self.margin = margin
        self.lambda_pt = lambda_pt
        self.loss_fn = loss_fn

    def pullaway_term(self, embedding):
        normalized_emb = F.normalize(embedding)
        similarity = torch.matmul(normalized_emb, normalized_emb.T).pow(2)
        batch_size = embedding.size(0)
        loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return loss_pt

    def discriminator_loss(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        batch_size = x.size(0)
        device = x.device
        latent_dim = self.generator.latent_dim
        #
        # 1. get discriminator loss from real data
        #
        x_recon, _ = self.discriminator(x)
        real_D_loss = self.loss_fn(x, x_recon)
        #
        # 2. get discriminator loss from fake data
        #
        z = torch.randn((batch_size, latent_dim)).to(device)
        fake_data = self.generator(z)
        fake_recon, _ = self.discriminator(fake_data)

        fake_D_loss = self.loss_fn(fake_data, fake_recon)
        fake_D_loss = F.relu(self.margin - fake_D_loss)

        return real_D_loss, fake_D_loss

    def generator_loss(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        latent_dim = self.generator.latent_dim

        z = torch.randn((batch_size, latent_dim)).to(device)
        fake_data = self.generator(z)
        fake_recon, embedding = self.discriminator(fake_data)

        pt = self.pullaway_term(embedding.squeeze())
        G_loss = self.loss_fn(fake_data, fake_recon) + self.lambda_pt * pt

        return G_loss