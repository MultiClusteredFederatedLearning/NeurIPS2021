import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MnistDataset(Dataset):
    def __init__(self, images, labels, img_size):
        self.images = images
        self.labels = labels.astype(int)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index].reshape(28, 28), mode='L')
        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)