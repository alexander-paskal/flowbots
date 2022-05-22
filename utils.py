import os
import torch.nn as nn
import torch
import json
from torch.utils.data import DataLoader
from losses import epe_loss, f1_all
from models import lookup


class HardwareManager:

    dtype = torch.float16
    use_gpu = True

    @classmethod
    def get_device(cls):
        if cls.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return device

    @classmethod
    def get_dtype(cls):
        return cls.dtype


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


def train(model, optimizer, loader, epochs=1, print_every=100, val_loader=None, val_every=10):
    """
    Ochestrates the train loop
    :return:
    """

    device = HardwareManager.get_device()
    dtype = HardwareManager.get_dtype()

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    losses = []


    for e in range(epochs):
        for t, (x, y) in enumerate(loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

            pred = model(x)
            loss = epe_loss(pred, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                print()


def evaluate(loader, model):
    """
    Evaluates a model on a dataset
    :param loader:
    :param model_fn:
    :param params:
    :return:
    """
    device = HardwareManager.get_device()
    dtype = HardwareManager.get_dtype()

    losses = []
    f1_ratios = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            pred = model(x)

            loss = epe_loss(pred, y).item()
            losses.append(loss)

            f1_ratio = f1_all(pred, y).item()
            f1_ratios.append(f1_ratio)

        avg_loss = torch.tensor(losses).mean().item()
        avg_percent = torch.tensor(f1_ratios).mean().item()
        print(f"Eval Results:")
        print(f"EPE Loss: {round(avg_loss, 3)}, F1_all Error: %{round(avg_percent, 3) * 100}")

    return {"epe": avg_loss, "f1_all": f1_ratio}


PARAMETERS_DIR = "parameters"
def save_model(model, name):
    """
    Saves the parameters of the model
    :param model:
    :param name:
    :return:
    """
    if not os.path.exists(PARAMETERS_DIR):
        os.mkdir(PARAMETERS_DIR)

    torch.save(model, os.path.join(PARAMETERS_DIR, name + ".pth"))

    obj = {
        "model": model.title
    }

    with open(os.path.join(PARAMETERS_DIR, name + ".json"), "w") as f:
        json.dump(obj, f)


def load_model(name):
    """

    :param name:
    :return:
    """

    with open(os.path.join(PARAMETERS_DIR, name + ".json")) as f:
        obj = json.load(f)

    model_title = obj["model"]
    model_cls = lookup[model_title]
    model = model_cls()

    model.load_state_dict(torch.load(os.path.join(PARAMETERS_DIR, name + ".pth")))

    return model



if __name__ == '__main__':
    from datasets import flying_chairs

    fc = flying_chairs(split="val")
    dl = DataLoader(fc, 1, True)





