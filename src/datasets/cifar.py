import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CIFARDataset(Dataset):
    def __init__(self, images, labels, training=True):
        self.images = images
        self.labels = labels
        if training:
            self.transform = self._train_transform()
        else:
            self.transform = self._test_transform()
    
    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def _train_transform(self):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        return transform
    
    def _test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        return transform

    def __len__(self):
        return len(self.images)


class BYOLCIFARDataset(CIFARDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_aug = self._train_transform()
        self.test_aug = self._test_transform()

    
    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        x = self.test_aug(img.copy())
        x1 = self.train_aug(img.copy())
        x2 = self.train_aug(img.copy())
        target = self.labels[index]
        return x, x1, x2, target