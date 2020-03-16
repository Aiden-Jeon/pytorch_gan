from layers import ConvLayer, ConvTransposeLayer


def make_generator_layers(dataset: str, latent_dim: int = 100, hidden_channel: int = 128) -> list:
    if dataset in ["mnist", "fmnist"]:
        layers = [
            ConvTransposeLayer(latent_dim, hidden_channel * 4, 4, 1, 0, True, "relu"),
            ConvTransposeLayer(hidden_channel * 4, hidden_channel * 2, 4, 1, 0, True, "relu"),
            ConvTransposeLayer(hidden_channel * 2, hidden_channel, 4, 2, 1, True, "relu"),
            ConvTransposeLayer(hidden_channel, 1, 4, 2, 1, False, "tanh"),
        ]
    elif dataset in ["lsun"]:
        layers = [
            ConvTransposeLayer(latent_dim, hidden_channel * 8, 4, 1, 0, True, "relu"),
            ConvTransposeLayer(hidden_channel * 8, hidden_channel * 4, 4, 2, 1, True, "relu"),
            ConvTransposeLayer(hidden_channel * 4, hidden_channel * 2, 4, 2, 1, True, "relu"),
            ConvTransposeLayer(hidden_channel * 2, hidden_channel, 4, 2, 1, True, "relu"),
            ConvTransposeLayer(hidden_channel, 3, 4, 2, 1, False, "tanh"),
        ]
    else:
        raise ValueError
    return layers


def make_discriminator_layers(dataset: str, hidden_channel: int = 128) -> list:
    if dataset in ["mnist", "fmnist"]:
        layers = [
            ConvLayer(1, hidden_channel, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel, hidden_channel * 2, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel * 2, hidden_channel * 4, 4, 1, 0, True, "leakyrelu"),
            ConvLayer(hidden_channel * 4, 1, 4, 1, 0, False, "sigmoid"),
        ]
    elif dataset in ["lsun"]:
        layers = [
            ConvLayer(3, hidden_channel, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel, hidden_channel * 2, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel * 2, hidden_channel * 4, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel * 4, hidden_channel * 8, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel * 8, 1, 4, 1, 0, False, "sigmoid"),
        ]
    else:
        raise ValueError
    return layers