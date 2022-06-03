import os
import torch.nn as nn
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from losses import epe_loss, f1_all
from models import lookup
import time


class HardwareManager:

    use_gpu = True

    @classmethod
    def get_device(cls):
        if cls.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return device

    @classmethod
    @property
    def dtype(cls):
        if cls.use_gpu and torch.cuda.is_available():
            # dtype = torch.float16
            dtype = torch.float32
        else:
            dtype = torch.float32
        return dtype


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


def train(model, optimizer, loader, epochs=1, print_every=1, grad_accum=1, val_loader=None, val_every=4, verbose=False):
    """
    Ochestrates the train loop
    :return:
    """

    device = HardwareManager.get_device()
    dtype = HardwareManager.dtype

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    best_loss = np.inf
    best_params = model.state_dict()

    # change to MSEloss
    mse_loss = nn.MSELoss()

    for e in range(epochs):
        print(f"Epoch {e+1}")


        # yield model, "hello", "there"
        # import time
        # time.sleep(5)
        # continue


        e_losses = []
        s = time.time()
        for t, (x, y) in enumerate(loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

            pred = model(x)
            # loss = epe_loss(pred, y)
            loss = mse_loss(pred.flatten(), y.flatten())
            loss.backward()

            if verbose:
                if t % 10 == 0:
                    print(f"Iteration {t}, loss {loss}")

            if t % grad_accum == 0:  # for gradient accumulation
                optimizer.step()
                optimizer.zero_grad()

            e_losses.append(loss.item())
        avg_e_loss = sum(e_losses) / len(e_losses)


        # checkpointing
        if avg_e_loss < best_loss:
            best_loss = avg_e_loss
            best_params = model.state_dict()

        if e % print_every == 0:
            print('Epoch {}, Loss = {:.3f}, train time = {:.3f} seconds'.format(e, avg_e_loss, time.time() - s))
            print()
        if val_loader is not None and e % val_every == 0:
            val_results = evaluate(val_loader, model)
            print(f"Validation Results: {val_results}")

        else:
            val_results = None

        cur_params = model.state_dict()  # save current parameters

        model.load_state_dict(best_params)  # load and yield the best model parameters
        yield model, avg_e_loss, val_results
        model.load_state_dict(cur_params)


def evaluate(loader, model):
    """
    Evaluates a model on a dataset
    :param loader:
    :param model_fn:
    :param params:
    :return:
    """
    device = HardwareManager.get_device()
    dtype = HardwareManager.dtype

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


PARAMETERS_DIR = "weights"
def save_model(model, name, info):
    """
    Saves the parameters of the model
    :param model:
    :param name:
    :return:
    """
    if not os.path.exists(PARAMETERS_DIR):
        os.mkdir(PARAMETERS_DIR)

    torch.save(model.state_dict(), os.path.join(PARAMETERS_DIR, name + ".pth"))

    obj = {
        "architecture": model.title
    }
    obj.update(info)

    with open(os.path.join(PARAMETERS_DIR, name + ".json"), "w") as f:
        json.dump(obj, f)


def load_model(name):
    """

    :param name:
    :return:
    """

    with open(os.path.join(PARAMETERS_DIR, name + ".json")) as f:
        info = json.load(f)

    model_title = info["architecture"]
    model_cls = lookup[model_title]
    model = model_cls()

    model.load_state_dict(torch.load(os.path.join(PARAMETERS_DIR, name + ".pth"), map_location=torch.device("cpu")))

    return model, info


if __name__ == '__main__':
    from datasets import flying_chairs

    fc = flying_chairs(split="val")
    dl = DataLoader(fc, 1, True)





