import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FemnistDataset(Dataset):
    def __init__(self, images, labels, training=True):
        self.images = images
        self.labels = labels.astype(int)
        if training:
            self.transform = self._train_transform()
        else:
            self.transform = self._test_transform()

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)

    def _train_transform(self):
        transform = transforms.Compose([
                    transforms.RandomAffine(45, (0.15, 0.15)),
                    transforms.RandomResizedCrop(28, (0.75, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, ), (0.5, ))
                ])
        return transform
    
    def _test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        return transform

class BYOLFemnistDataset(FemnistDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_aug = self._train_transform()
        self.test_aug = self._test_transform()
    
    def __getitem__(self, index):
        img = Image.fromarray(self.images[index], 'L')
        x = self.test_aug(img.copy())
        x1 = self.train_aug(img.copy())
        x2 = self.train_aug(img.copy())
        target = self.labels[index]
        return x, x1, x2, target

if __name__ == "__main__":
    pass
