from utils import evaluate, load_model, HardwareManager
from datasets import flying_chairs, sintel, hd1k
from torch.utils.data import DataLoader
import torch

HardwareManager.use_gpu = True
# dataset = flying_chairs(split="val")
dataset = sintel(pass_name="clean")
# dataset = hd1k()
# model, info = load_model("flownetSSwarped-0609")
model, info = load_model("simple_s_test_new_epe")
print("Dataset length:",len(dataset))
dloader = DataLoader(dataset, batch_size=10)

with torch.no_grad():
    evaluate(dloader, model, verbose=True, batches=50)
