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

    # args = parser.parse_args()
    class args:
        epochs = 10
        grad_accum = 1
        model_name = "simple_s_test_newest_loss"
        verbose = 'true'
        learning_rate = 0.001
        batch_size = 10
        architecture = 'flownet-s'


    HardwareManager.use_gpu = False

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
    try:
        model_unstacked, info = load_model(MODEL_NAME)
        losses = info["losses"]
        validations = info["validations"]
        epochs_trained = info["epochs_trained"]
    
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Could not locate model, initializing new.")
    
        if ARCHITECTURE != 'flownet-s':
            print("architecture must be flownet. Exiting.")
            sys.exit(-1)
    
        model_cls = lookup[ARCHITECTURE]
        model_unstacked = initialize(model_cls)
   
    
    model = FlownetStacked(model_unstacked, FlowNetS, warping=True, frozen=[True, False])

    losses = []
    validations = []
    epochs_trained = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Training")
    for model, train_loss, train_validation in train(
            model, optimizer, train_loader, epochs=EPOCHS, grad_accum=GRAD_ACCUM, verbose=VERBOSE, val_loader=val_loader,
    print_loss=1):
        losses.append(train_loss)
        validations.append(train_validation)
        epochs_trained += 1

        info = {
            "losses": losses,
            "validations": validations,
            "epochs_trained": epochs_trained,
        }

        # save_model(model, MODEL_NAME, info=info)

    print("Training completed")



