from utils import evaluate, load_model
from datasets import flying_chairs, sintel
from torch.utils.data import DataLoader
import torch


dataset = sintel(pass_name="final")
model, info = load_model("flownet-ss-first-0607")
print("Dataset length:",len(dataset))
dloader = DataLoader(dataset, batch_size=10)

with torch.no_grad():
    evaluate(dloader, model, verbose=True)
