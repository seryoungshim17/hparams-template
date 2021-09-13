from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch

import argparse

def main(**kwargs):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = FashionMNIST(
        root="./data", 
        train=True, 
        transform=transform, 
        download=True
    )
    val_dataset = FashionMNIST(
        root="./data", 
        train=False, 
        transform=transform, 
        download=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperParameter")
#     parser.add_argument(
#         "--PATH",
#         type = str,
#         help = "Base Path",
#         default = "input/"
#     )
    parser.add_argument(
        "--OPTIMIZER",
        type = str,
        help = "Optimizer(Adam/SGD)",
        default = "Adam"
    )
    parser.add_argument(
        "--LR",
        type = float,
        help = "Learning rate",
        default = 1e-5
    )
    parser.add_argument(
        "--EPOCH",
        type = int,
        help = "Total Training Epoch",
        default = 20
    )
    parser.add_argument(
        "--BATCH_SIZE",
        type = int,
        help = "Batch Size",
        default = 128
    )
    arg = parser.parse_args()
    main(
        optim=arg.OPTIMIZER,
        learning_rate=arg.LR,
        epoch=arg.EPOCH,
        batch_size=arg.BATCH_SIZE
    )