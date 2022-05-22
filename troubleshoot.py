from models.flownetS import FlowNetS
import torch
from torch.utils.data import DataLoader
from datasets import flying_chairs


train_dataset = flying_chairs(split="train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = flying_chairs(split="val")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


from utils import HardwareManager, train, initialize, save_model
HardwareManager.dtype = torch.float32

model = initialize(FlowNetS)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model, results = train(model, optimizer, train_loader, epochs=50)


save_model(model, "test-s")