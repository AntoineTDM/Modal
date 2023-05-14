from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# Not sure how to call the images yet. For now probably have separate dataloader?
# We will need to cap the number of unlabelled images used by the way
# Store the names and use them to build a dataloader?


class CustomDataset(Dataset):
    def __init__(self, dataset, transform = None):
        self.unlabelled_dataset_path = dataset
        self.transform = transform
        images_list = os.listdir(self.unlabelled_dataset_path)
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.unlabelled_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.transform(image)
        return image


class DataModule:
    def __init__(
        self,
        unlabelled_dataset_path,
        train_transform,
        batch_size,
        num_workers,
    ):
        self.unlabelled_dataset = CustomDataset(unlabelled_dataset_path, transform=train_transform)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def unlabelled_dataloader(self):
        return DataLoader(
            self.unlabelled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )



