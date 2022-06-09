from models import FlownetStacked, FlowNetS
import torch
from utils import load_model, save_model
from models import lookup
import torch
from torch.utils.data import DataLoader
from datasets import flying_chairs
from utils import HardwareManager, train, initialize, save_model, load_model
import sys
import math
import argparse
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--model-name', '-m', type=str,
                        help='the name of the model to be trained. Distinct from the model architecture:'
                             ' choose a unique name')
    # parser.add_argument('--dataset', '-d', help='the dataset being trained on')
    parser.add_argument('--epochs', '-e', type=int, help='the number of epochs to train for')
    parser.add_argument('--architecture', '-a', default=None, help='the model architecture')
    parser.add_argument('--batch-size', '-b', default=50, type=int, help='the minibatch size')
    parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float, help='the learning rate to be used')
    parser.add_argument('--verbose', '-v', default='true', help='prints out the loss every 10 minibatches')
    parser.add_argument('--grad-accum', '-g', default=1, type=int, help='The gradient accumulation rate. Default is 1')

    #args = parser.parse_args()
    class args:
        epochs = 10
        grad_accum = 1
        model_name = "0"
        verbose = 'true'
        learning_rate = 0.001
        batch_size = 30
        architecture = 'flownet-s'


    EPOCHS = args.epochs
    GRAD_ACCUM = args.grad_accum
    MODEL_NAME = args.model_name
    ARCHITECTURE = args.architecture
    VERBOSE = True if args.verbose == "true" else False
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size

    print("loading dataset")
    train_dataset = flying_chairs(split="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = flying_chairs(split="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print(f"Iterations per training epoch: {int(math.ceil(len(train_dataset) / BATCH_SIZE))}")

    print(f"Training on: {HardwareManager.get_device()}")

    print("instantiating model")
    model_unstacked_1, unused_info = load_model('FlowNetC_FlyingChairs_scheduler_long')
    model_unstacked_2, unused_info = load_model('flownet-ss-first-0607')
    # params = list(model_unstacked_2.parameters())
    # new_params = []
    # i=0
    # for name, p in model_unstacked_2.named_parameters():
    #     if i>61:
    #         new_params.append(p)
    #     i=i+1
    # model_unstacked_2_real = nn.Sequential(new_params)
    # #model_cls = lookup['flownet-s']
    # #model_unstacked_2 = initialize(model_cls, in_channels=8)
    model_cls = lookup['flownet-s']
    model_unstacked_3 = initialize(model_cls, in_channels=8)

    # print(model_unstacked_1)
    
    # i=0
    # for name, param in model_unstacked_2.named_parameters():
    #     print(name)
    #     print(i)
    #     i=i+1
        
    model = FlownetStacked(model_unstacked_1, model_unstacked_2.net2, model_unstacked_3, warping=False, frozen=[True, False, False])
    #    


    try:
        trash, info = load_model(MODEL_NAME)
        model.load_state_dict(torch.load('weights\\CSS_test.pth'))
        losses = info["losses"]
        validations = info["validations"]
        epochs_trained = info["epochs_trained"]
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Could not locate model, initializing new.")
        losses = []
        validations = []
        epochs_trained = 0


   
    
    #Loop set to freeze weights
    i=0
    for name, param in model.named_parameters():
        i=i+1
        if i <= 54:
            param.requires_grad = False

    print('saving..')
    save_model(model, MODEL_NAME, info=info)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Training")
    for model, train_loss, train_validation in train(
            model, optimizer, train_loader, epochs=EPOCHS, grad_accum=GRAD_ACCUM, verbose=VERBOSE, val_loader=val_loader):
        losses.append(train_loss)
        validations.append(train_validation)
        epochs_trained += 1

        info = {
            "losses": losses,
            "validations": validations,
            "epochs_trained": epochs_trained,
        }

        save_model(model, MODEL_NAME, info=info)

    print("Training completed")



