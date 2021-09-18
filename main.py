from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch

import argparse
from model import Resnet18Model
from train import Train

def main(**kwargs):
    print(f'[#{kwargs["exp_num"]}] OPTIMIZER: {kwargs["optim"]} | LEARNING_RAGE: {kwargs["learning_rate"]}')
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = FashionMNIST(
        root="./data", 
        train=True, 
        transform=transform, 
        download=False
    )
    val_dataset = FashionMNIST(
        root="./data", 
        train=False, 
        transform=transform, 
        download=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=2
    )
    model = Resnet18Model(num_classes=10).to(DEVICE)
    
    model_train = Train(
        model=model,
        num_classes=10, 
        trainloader=train_loader, 
        valloader=val_loader
    )
    best_weights = model_train.train(
        optim=kwargs["optim"],  
        device=DEVICE,
        learning_rate=kwargs["learning_rate"], 
        epochs=kwargs["epoch"]
    )
    
    # Tensorboard Hparams
    hp_writer = SummaryWriter('logs/hp_tunning/')

    hp_writer.add_hparams(
        {
            "optimizer" : kwargs["optim"],
            "learning_rate" : kwargs["learning_rate"]
        },
        {
            "loss": best_weights["loss"],
            "acc" : best_weights["acc"],
            "f1": best_weights["f1"]
        },
        run_name = f'exp_{arg.EXP_NUM}'
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
        default = 64
    )
    parser.add_argument(
        "--EXP_NUM",
        type = int,
        help = "Experiment Number",
        default = 1
    )
    arg = parser.parse_args()
    main(
        optim=arg.OPTIMIZER,
        learning_rate=arg.LR,
        epoch=arg.EPOCH,
        batch_size=arg.BATCH_SIZE,
        exp_num = arg.EXP_NUM
    )