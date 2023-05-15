from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,ConcatDataset,Dataset
import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # labeled image
            image, label = self.dataset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        else:
            # unlabeled image
            image_path = self.dataset[idx] # name of image
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            return image

# Not sure how to call the images yet. For now probably have separate dataloader?
# We will need to cap the number of unlabelled images used by the way
# Store the names and use them to build a dataloader?


def _list_images(path):
    images_list = os.listdir(path)[:1000] # Don't want too many at first
    return [os.path.join(path, image) for image in images_list if image.endswith(".jpg")]


class DataModule:
    def __init__(
        self,
        train_dataset_path,
        unlabelled_dataset_path,
        train_transform,
        val_transform,
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
        self.TF = [augment_transform1, augment_transform2, augment_transform3, augment_transform4, augment_transform5]
        self.augment_dataset = [CustomDataset(self.train_dataset, transform=transform) for transform in self.TF]
        self.augmented_dataset = ConcatDataset(self.augment_dataset + [self.train_dataset])
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.val_dataset.transform = val_transform
        self.unlabelled_dataset = CustomDataset(_list_images(unlabelled_dataset_path), transform=train_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def augmented_dataloader(self):
        return DataLoader(
            self.augmented_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def unlabelled_dataloader(self):
        return DataLoader(
            self.unlabelled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def add_to_training_set(self, model, threshold=0.9):
        model.eval()
        with torch.no_grad():
            for images in self.unlabelled_dataloader():
                predictions = torch.softmax(model(images[0]), dim=1)
                max_probs, max_classes = torch.max(predictions, dim=1)
                for i, max_prob in enumerate(max_probs):
                    if max_prob >= threshold:
                        self.augmented_dataset.dataset.imgs.append((images[i], max_classes[i]))
        # self.augmented_dataset = ConcatDataset(
        #     [
        #         self.train_dataset,
        #         self.augment_dataset1,
        #         self.augment_dataset2,
        #         self.augment_dataset3,
        #         self.augment_dataset4,
        #         self.augment_dataset5,
        #     ]
        # )
