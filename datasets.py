from torch.utils.data import Dataset
from read_slide import create_tile_dataset
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
import numpy as np
import random
from torchvision import transforms
from utils import decompress_directory, remove_directory
from torch.utils.data import Dataset, Sampler
from typing import Iterator


def load_slide(slides_path, slide):
    print(slide)
    decompress_directory(slides_path + f"/{slide}.tar.gz", slides_path)
    slide_tiles = create_tile_dataset(slide, slides_path)
    remove_directory(slides_path + f"/{slide}")
    return slide_tiles


class SimCLRDataAugmentation:
    def __init__(self):
        # Define augmentation pipeline
        self.augmentation1 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),  # Horizontal flip
                transforms.GaussianBlur(
                    kernel_size=7, sigma=(0.1, 2.0)
                ),  # Gaussian blur
                transforms.ToTensor(),  # Convert to tensor
            ]
        )
        self.jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)

    def random_rotation(self, image):
        """Apply random rotation (0, 90, 180, 270 degrees)."""
        degrees = random.choice([0, 90, 180, 270])  # Choose one of the four angles
        return rotate(image, degrees)

    def add_random_noise(self, image):
        """Add random white noise to the image tensor."""
        noise = torch.randn_like(image) * 0.02  # Adjust the noise intensity as needed
        return image + noise

    def __call__(self, image):
        # Apply augmentations to generate two views
        view1 = self.augmentation1(image)
        view2 = self.augmentation1(image)
        for i in range(4):
            view1[i : i + 1] = self.jitter(view1[i : i + 1])
            view2[i : i + 1] = self.jitter(view2[i : i + 1])
        view1 = self.random_rotation(view1)
        view2 = self.random_rotation(view2)
        view1 = self.add_random_noise(view1)
        view2 = self.add_random_noise(view2)
        return view1, view2


class Custom4ChannelDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data: Tensor of shape (N, 4, 32, 32), where N is the number of images.
            transform: Augmentation pipeline.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx])
        if self.transform:
            view1, view2 = self.transform(image)
        else:
            view1, view2 = torch.tensor(image), torch.tensor(image)
        return (view1.float() / 255.0, view2.float() / 255.0)


def loading_datasets(rarest_path, slides_path, slide_list):
    data_normal = []
    for slide in slide_list:
        data_normal.append(load_slide(slides_path, slide))
    data_normal = np.concatenate(data_normal, axis=0)
    data_rare = np.load(rarest_path)
    idxs_normal = np.random.permutation(data_normal.shape[0])
    idxs_rare = np.random.permutation(data_rare.shape[0])
    dataset_normal = Custom4ChannelDataset(
        data_normal[idxs_normal[: int(data_normal.shape[0] * 0.8)]],
        transform=SimCLRDataAugmentation(),
    )
    dataset_rare = Custom4ChannelDataset(
        data_rare[idxs_rare[: int(data_rare.shape[0] * 0.8)]],
        transform=SimCLRDataAugmentation(),
    )
    dataset_normal_val = Custom4ChannelDataset(
        data_normal[idxs_normal[int(data_normal.shape[0] * 0.8) :]],
        transform=SimCLRDataAugmentation(),
    )
    dataset_rare_val = Custom4ChannelDataset(
        data_rare[idxs_rare[int(data_rare.shape[0] * 0.8) :]],
        transform=SimCLRDataAugmentation(),
    )
    return dataset_normal, dataset_rare, dataset_normal_val, dataset_rare_val

class CustomDataLoader:
    def __init__(
        self, 
        dataset1: Dataset,
        dataset2: Dataset, 
        batch_size: int, 
        shuffle: bool = False, 
        drop_remainder: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        """
        Custom DataLoader for PyTorch.

        Args:
            dataset (Dataset): The dataset to load from.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            drop_remainder (bool): Whether to drop the last batch if incomplete.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.dtype = dtype
        self.sampler = self._create_sampler()

    def _create_sampler(self) -> Sampler:
        """Creates a sampler for the dataset."""
        indices1 = list(range(len(self.dataset1)))
        indices2 = list(range(len(self.dataset2)))
        if self.shuffle:
            indices1 = torch.randperm(len(self.dataset1)).tolist()
            indices2 = torch.randperm(len(self.dataset2)).tolist()
        return indices1, indices2

    def _batch_sampler(self) -> Iterator:
        """Yields batches of indices."""
        min_sampler_len = min(len(self.sampler[0]), len(self.sampler[1]))
        for i in range(0, min_sampler_len, self.batch_size):
            if i + self.batch_size > min_sampler_len:
                if self.drop_remainder:
                    break
            yield self.sampler[0][i:i + self.batch_size], self.sampler[1][i:i + self.batch_size]

    def __iter__(self) -> Iterator:
        """Resets the iteration and starts a new one."""
        self.sampler = self._create_sampler()  # Shuffle or reset indices
        self.batches = list(self._batch_sampler())
        self.batch_iter = iter(self.batches)
        return self

    def __next__(self):
        """Returns the next batch."""
        try:
            batch_indices = next(self.batch_iter)
            return (torch.tensor([self.dataset1[i] for i in batch_indices[0]], dtype=self.dtype),
                    torch.tensor([self.dataset2[i] for i in batch_indices[1]], dtype=self.dtype)
            )
        except StopIteration:
            raise StopIteration

    def __len__(self):
        """Returns the number of batches."""
        return len(self.batches)
