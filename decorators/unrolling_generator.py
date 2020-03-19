def unrolling_generator(get_generator_loss):
    from copy import deepcopy
    def unrolling(self, x):
        if self.unrolling_step > 0:
            # backup discriminator
            # : because of optimizer we will not use backup_discriminator
            backup_discriminator = deepcopy(self.discriminator)

            # update discriminator unrolling k step
            device = next(self.generator.parameters()).device
            x = x.to(device)
            for _ in range(self.unrolling_step):
                self.discriminator_opt.zero_grad()
                real_D_loss, fake_D_loss = self.get_discriminator_loss(x)
                D_loss = real_D_loss + fake_D_loss
                D_loss.backward()
                self.discriminator_opt.step()
            G_loss = get_generator_loss(self, x)

            # backup the discriminator and delete
            self.discriminator = backup_discriminator
            del backup_discriminator
        else:
            G_loss = get_generator_loss(self, x)
        return G_loss
    return unrolling