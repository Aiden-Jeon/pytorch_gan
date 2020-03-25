from layers import ConvLayer, ConvTransposeLayer


def make_generator_layers(
    input_size: int, latent_dim: int = 100, hidden_channel: int = 128, last_act: str = "sigmoid"
) -> list:
    # mnist, fmnist
    if input_size == 28:
        layers = [
            ConvTransposeLayer(latent_dim, hidden_channel * 4, 4, 1, 0, True, "relu"),
            ConvTransposeLayer(hidden_channel * 4, hidden_channel * 2, 4, 1, 0, True, "relu"),
            ConvTransposeLayer(hidden_channel * 2, hidden_channel, 4, 2, 1, True, "relu"),
            ConvTransposeLayer(hidden_channel, 1, 4, 2, 1, False, last_act),
        ]
    # cifar10
    elif input_size == 32:
        layers = [
            ConvTransposeLayer(latent_dim, hidden_channel * 4, 4, 1, 0, True, "relu"),
            ConvTransposeLayer(hidden_channel * 4, hidden_channel * 2, 4, 2, 1, True, "relu"),
            ConvTransposeLayer(hidden_channel * 2, hidden_channel, 4, 2, 1, True, "relu"),
            ConvTransposeLayer(hidden_channel, 3, 4, 2, 1, False, last_act),
        ]
    else:
        raise ValueError
    return layers


def make_discriminator_layers(input_size: int, hidden_channel: int = 128, last_act: str = "sigmoid") -> list:
    # mnist, fmnist
    if input_size == 28:
        layers = [
            ConvLayer(1, hidden_channel, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel, hidden_channel * 2, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel * 2, hidden_channel * 4, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel * 4, 1, 3, 1, 0, False, last_act),
        ]
    # cifar10
    elif input_size == 32:
        layers = [
            ConvLayer(3, hidden_channel, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel, hidden_channel * 2, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel * 2, hidden_channel * 4, 4, 2, 1, True, "leakyrelu"),
            ConvLayer(hidden_channel * 4, 1, 4, 1, 0, False, last_act),
        ]
    else:
        raise ValueError
    return layers