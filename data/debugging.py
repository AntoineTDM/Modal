from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate
import torch


# class Debugging:
#     def __init__(
#         self,
#         test_dataset_path,
#         test_transform,
#         batch_size,
#         num_workers,
#     ):
#         self.test_dataset = ImageFolder(test_dataset_path, transform=test_transform)
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#
#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )

class Debugging:
    def __init__(
        self,
        test_dataset_path,
        test_transform,
        batch_size,
        num_workers,
    ):
        self.test_dataset = ImageFolder(test_dataset_path, transform=test_transform)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=150,
            shuffle=False,
            num_workers=self.num_workers,
        )