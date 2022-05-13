import torch.nn as nn


def initialize(model_cls, *args, weights="xavier", **kwargs):
    """
    Factory method for a neural network. Takes care of weight initialization, etc.
    :param cls: the network class to be initialized
    :return:
    """

    model = model_cls(*args, **kwargs)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.uniform_(m.bias)
            nn.init.xavier_uniform_(m.weight)

        if isinstance(m, nn.ConvTranspose2d):
            if m.bias is not None:
                nn.init.uniform_(m.bias)
            nn.init.xavier_uniform_(m.weight)

    return model