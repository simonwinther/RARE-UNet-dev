from sys import stderr
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from data.data_manager import DataManager
from torchinfo import summary

# from trainer import Trainer
from data.datasets import (
    MedicalDecathlonDataset,
    ModalitiesDataset,
    VALID_TASKS,
    PreprocessedMedicalDecathlonDataset,
)
from trainers.trainer_factory import get_trainer
from utils.assertions import ensure_has_attr, ensure_has_attrs
from utils.losses import get_loss_fn
from utils.utils import RunManager, setup_environment, setup_logging
import random
import numpy as np
from utils.wandb_logger import get_wandb_logger
import argparse
import logging
from hydra.utils import instantiate as inst


EXCLUDED_TASKS = {"Task01_BrainTumour", "Task05_Prostate"}
DATASET_MAPPING = {
    task: PreprocessedMedicalDecathlonDataset for task in VALID_TASKS - EXCLUDED_TASKS
}
DATASET_MAPPING["Task01_BrainTumour"] = PreprocessedMedicalDecathlonDataset
DATASET_MAPPING["Task05_Prostate"] = PreprocessedMedicalDecathlonDataset
import os


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ensure_has_attrs(cfg, ["gpu"], Exception)

    setup_environment(cfg)

    task_name = cfg.dataset.name

    assert task_name in DATASET_MAPPING, f"Unknown dataset: {task_name}"

    run_manager = RunManager(cfg)
    setup_logging(cfg.get("logging", {}), run_manager.run_exp_dir)

    logger = logging.getLogger(__name__)
    gpu_device = cfg.gpu.devices[0]  # TODO: Handle multiple GPUs
    device = (
        torch.device(f"cuda:{gpu_device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    try:
        model = inst(cfg.architecture.path, cfg)
        optimizer = inst(cfg.training.optimizer.params, params=model.parameters())
        criterion = inst(cfg.training.loss.params)
        lr_scheduler = inst(cfg.training.scheduler.params, optimizer=optimizer)
    except Exception as e:
        logger.error(f"Failed to instantiate network: {e}")
        exit(1)
    
    #delete
    def count_unet_params(model, scale: int = 0):
        modules = (
            list(model.encoders)
            + list(model.pools)
            + list(model.enc_dropouts)
            + [model.bn]
            + list(model.up_convs)
            + list(model.decoders)
            + list(model.dec_dropouts)
            + [model.final_conv]
        )
        return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in UNet3D: {count_unet_params(model)}")
    #delete

    model.to(device)
    model_summary = summary(
        model,
        input_size=(
            cfg.training.batch_size,
            cfg.dataset.in_channels,
            *cfg.dataset.target_shape
        ),
        col_names=tuple(cfg.architecture.summary.col_names),
        verbose=cfg.architecture.summary.verbose,
        # optional tweaks to get closer to Keras style:
    )
    
    seed = cfg.get("seed", 42)
    dataset_class = DATASET_MAPPING[task_name]
    data_manager = DataManager(dataset_class, cfg, seed, tr_split_ratios=(0.80, 0.05))
    train_dataloader, val_dataloader = data_manager.get_dataloaders()

    wandb_logger = get_wandb_logger(cfg=cfg, model=model, model_summary=model_summary) 
    
    trainer = get_trainer(
        cfg,
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        run_manager,
        wandb_logger,
    )

    final_status_code = 0
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        final_status_code = 1
    finally:
        if wandb_logger:
            wandb_logger.finalize(exit_code=final_status_code)
        logger.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Image Segmentation Training Script"
    )

    main()
