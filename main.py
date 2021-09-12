from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch

def main():
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
    main()