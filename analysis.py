from utils import evaluate, load_model, HardwareManager
from datasets import flying_chairs, sintel, hd1k
from torch.utils.data import DataLoader
import torch

HardwareManager.use_gpu = True
dataset = hd1k()
model, info = load_model("flownet-ss-first-0607")
print("Dataset length:",len(dataset))
dloader = DataLoader(dataset, batch_size=10)

with torch.no_grad():
    evaluate(dloader, model, verbose=True)
