import os
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
        Load images from a directory
    """
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.label_names= []
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                label_names = filename.split('_')[0]
                self.images.append(os.path.join(directory, filename))
                self.label_names.append(int(label_names))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.label_names[idx]
        return image, label


    