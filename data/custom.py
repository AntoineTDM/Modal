from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,ConcatDataset,Dataset
from hydra.utils import instantiate
import torch


class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class CustomDatasetWrapper:
    def __init__(
        self,
        train_dataset_path,
        train_transform,
        augment_transform1,
        augment_transform2,
        augment_transform3,
        augment_transform4,
        augment_transform5,
        batch_size,
        num_workers,
    ):
        # NB: images need to be in a folder. Image folder then labels them according to the folder number
        self.dataset = ImageFolder(train_dataset_path, transform=train_transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [
                int(0.8 * len(self.dataset)),
                len(self.dataset) - int(0.8 * len(self.dataset)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )
        self.augment_dataset1 = CustomDataset(self.train_dataset, transform=augment_transform1)
        self.augment_dataset2 = CustomDataset(self.train_dataset, transform=augment_transform2)
        self.augment_dataset3 = CustomDataset(self.train_dataset, transform=augment_transform3)
        self.augment_dataset4 = CustomDataset(self.train_dataset, transform=augment_transform4)
        self.augment_dataset5 = CustomDataset(self.train_dataset, transform=augment_transform5)
        self.augmented_dataset = ConcatDataset([self.train_dataset, self.augment_dataset1, self.augment_dataset2, self.augment_dataset3, self.augment_dataset4, self.augment_dataset5])
        self.batch_size = batch_size
        self.num_workers = num_workers

    def augmented_dataloader(self):
        return DataLoader(
            self.augmented_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

