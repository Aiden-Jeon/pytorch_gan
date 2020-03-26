def log_metric(engine):
    if not hasattr(engine, 'history'):
        engine.history = []
    engine.history += [engine.state.metrics]
    
def print_metric(engine):
    state = f"Epoch {engine.state.epoch} - "
    for key, value in engine.state.metrics.items():
        state += f"{key}: {value:.4f}, "
    print(state)

def print_img(engine, model, num_img: int = 64, save_prefix: str = None):
    import torch
    from torchvision.utils import make_grid
    from math import sqrt
    import matplotlib.pyplot as plt
    
    nrow = ncol = sqrt(num_img)
    if nrow * ncol < num_img:
        nrow += 1

    with torch.no_grad():
        model.generator.eval()
        device = next(model.generator.parameters()).device
        z = torch.randn((num_img, model.generator.latent_dim)).to(device)
        fake_data = model.generator(z).cpu()
        fake_data_grid = make_grid(fake_data, int(nrow))
        plt.imshow(fake_data_grid.permute(1, 2, 0))
        plt.show()
        
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}-{engine.state.epoch}")