import logging
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, random_split
import os
from data.datasets import MedicalDecathlonDataset
from typing import Tuple, Type
import numpy as np

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(
        self,
        dataset_class: Type[MedicalDecathlonDataset],
        cfg: DictConfig,
        seed: int,
        tr_split_ratios: Tuple[float, float] = (0.80, 0.05),
    ):
        """
        Initialize the DataManager with a dataset and configuration.
        """
        self.dataset_class = dataset_class
        self.cfg = cfg
        self.train_ratio, self.val_ratio = tr_split_ratios
        self.seed = seed
        self.dataset_subset_size = cfg.get("dataset_subset_size", None)
        self.train_dataset, self.val_dataset = self._create_datasets()
        self.train_dataloader, self.val_dataloader = self._create_dataloaders()

    def _create_datasets(self):
        base = self.cfg.dataset.base_path
        images_path = f"{base}{self.cfg.dataset.images_subdir}"
        masks_path = f"{base}{self.cfg.dataset.labels_subdir}"

        image_files_orig_sorted = sorted(os.listdir(images_path))
        mask_files_orig_sorted = sorted(os.listdir(masks_path))
        N = len(image_files_orig_sorted)

        if N == 0:
            raise ValueError("Dataset is empty. Cannot create datasets.")
        assert N == len(
            mask_files_orig_sorted
        ), "Mismatch between total images and masks!"

        import random

        indices = list(range(N))

        random.seed(self.seed)
        random.shuffle(indices)

        if self.dataset_subset_size is not None:
            total_samples = N
            pct = 100 * self.dataset_subset_size / total_samples
            logger.warning(
                f"Dataset subset enabled: loading only {self.dataset_subset_size}/{total_samples} "
                f"samples ({pct:.1f}% of full dataset) for faster Hyperparameter Optimization (HPO) runs. "
                "To use the entire dataset, set 'dataset_subset_size' to None in your config.")
            indices = indices[: self.dataset_subset_size]

        shuffled_image_files = [image_files_orig_sorted[i] for i in indices]
        shuffled_mask_files = [mask_files_orig_sorted[i] for i in indices]

        M = len(shuffled_image_files)
        n_trainval = int(self.train_ratio * M)
        trainval_imgs_pool = shuffled_image_files[:n_trainval]
        trainval_msks_pool = shuffled_mask_files[:n_trainval]

        test_imgs_dm_pool = shuffled_image_files[n_trainval:]

        n_val_pool_size = len(trainval_imgs_pool)
        n_val = int(self.val_ratio * n_val_pool_size)

        val_imgs = trainval_imgs_pool[:n_val]
        val_msks = trainval_msks_pool[:n_val]

        train_imgs = trainval_imgs_pool[n_val:]
        train_msks = trainval_msks_pool[n_val:]

        train_ds = self.dataset_class(
            self.cfg, phase="train", image_files=train_imgs, mask_files=train_msks
        )
        val_ds = self.dataset_class(
            self.cfg, phase="val", image_files=val_imgs, mask_files=val_msks
        )

        return train_ds, val_ds

    def _create_dataloaders(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            drop_last=self.cfg.architecture.drop_last,
            persistent_workers=self.cfg.training.persistent_workers,
            prefetch_factor=self.cfg.training.prefetch_factor,
        )
        # TODO maybe make validation instead of training... easy to do inside config/validation instead of config/training, but consider...
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            drop_last=False,
            persistent_workers=self.cfg.training.persistent_workers,
            prefetch_factor=self.cfg.training.prefetch_factor,
        )

        return train_dataloader, val_dataloader

    def get_dataloaders(self):
        """Return the train, val, and test DataLoaders."""
        return self.train_dataloader, self.val_dataloader
