"""
    Run the hw1
"""
import torch
from torch.utils.data import DataLoader

from utils.p1_dataloader import ImageDataset

print('-'*30)
print('Now start the program')
print('-'*30)

def get_device():
    """
        Check if CUDA or mac m1 GPU is available
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")
print("Now run the main program")
print('-'*30)


def main():
    """
        Main function to run the code of homework
    """
    dataset = ImageDataset(directory='hw1_data/p1_data/train_50', transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for images, labels in dataloader:
        print(images, labels)


if __name__ == '__main__':
    main()
