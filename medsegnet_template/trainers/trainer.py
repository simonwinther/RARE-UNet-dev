import time
from typing import Any, Dict, List, Optional, Tuple
from utils import metrics  # Assuming your metrics live here
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from omegaconf import DictConfig
from data.datasets import MedicalDecathlonDataset
from utils.assertions import ensure_in
from utils.metric_collecter import Agg, MetricCollector
from utils.metrics import dice_coefficient_classes, safe_mean
from utils.wandb_logger import WandBLogger
from utils.utils import RunManager
from trainers.callbacks.early_stopping import EarlyStopping
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils.table import print_train_val_table
import wandb
from wandb import Table
import logging

from torch.amp import GradScaler

from torch import autocast

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        run_manager: RunManager,
        wandb_logger: Optional[WandBLogger],
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        self.device = device
        self.rm = run_manager
        self.wandb_logger = wandb_logger

        self.logger = logging.getLogger(__name__)
        self.num_epochs = cfg.training.num_epochs
        self.num_classes = cfg.dataset.num_classes
        self.class_labels = {i: f"class_{i}" for i in range(self.num_classes)}
        self.ignore_index = cfg.dataset.get("ignore_index", 0)

        self.max_norm = self.cfg.training.get("grad_clip_norm", 1.0)
        # Setup early stopping if configured
        self.early_stopper = None
        early_cfg = cfg.training.get("early_stopper", None)
        if early_cfg:
            self.early_stopper = EarlyStopping(
                patience=early_cfg.get("patience", 15),
                delta=early_cfg.get("delta", 0.0),
                criterion=early_cfg.get("criterion", "loss"),
                verbose=early_cfg.get("verbose", True),
            )
        if not self.early_stopper:
            self.logger.info("No early stopping configured.")

        self.metrics = ["dice", "iou", "precision", "recall", "f1"]

        # Setup metric collector
        self.mc = MetricCollector()
        self._setup_metric_collector_rules()

        self.use_amp = cfg.get("use_amp", False) and self.device.type == "cuda"
        if self.use_amp:
            # device.type is either "cuda" or "cpu"
            self.scaler = torch.GradScaler(device.type)

            self.logger.info("Using mixed precision training with GradScaler.")

        self.logger.info(f"Initialized BaseTrainer for {self.num_epochs} epochs.")

    def _setup_metric_collector_rules(self):
        """Sets up the basic rules for the metric collector. Can be overridden."""
        self.mc.set_rule("loss", Agg.MEAN)
        for metric in self.metrics:
            self.mc.set_rule(f"{metric}_fullres", Agg.MEAN)
            self.mc.set_rule(f"{metric}_per_class_fullres", Agg.LIST_MEAN)
        self.logger.debug("Base metric collector rules set.")

    def train(self):
        start_time = time.time()

        for epoch in trange(self.num_epochs, desc="Training"):
            epoch_start_time = time.time()
            train_dict = self.train_one_epoch(epoch)
            train_dict["epoch_time"] = time.time() - epoch_start_time

            epoch_start_time = time.time()
            val_dict = self.validate(epoch)
            val_dict["epoch_time"] = time.time() - epoch_start_time

            # ----------- DELETE ME DEBUG -----------
            logger.info(f"Epoch Time: {train_dict['epoch_time']:.2f} seconds")
            dice_full_res = train_dict.get("dice_fullres", 0.0)
            dice_fullres = val_dict.get("dice_fullres", 0.0)
            logger.info(
                f"Train Dice Full Res: {dice_full_res:.4f}, Val Dice Full Res: {dice_fullres:.4f}"
            )
            # ----------- DELETE ME DEBUG -----------
            if self.lr_scheduler:
                if isinstance(
                    self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.lr_scheduler.step(val_dict["dice_fullres"])
                else:
                    self.lr_scheduler.step()
            train_dict["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            if self.wandb_logger:
                self.log_to_wandb(train_dict, step=epoch, prefix="train", commit=False)
                self.log_to_wandb(val_dict, step=epoch, prefix="val", commit=True)

            if self.cfg.logging.get("ppmat", False):
                table_str = print_train_val_table(train_dict, val_dict)
                tqdm.write(f"\nEpoch {epoch + 1} Results:\n{table_str}")

            if self.early_stopper:
                for key in ("loss", "dice_fullres"):
                    ensure_in(key, val_dict, KeyError)

                stop, improved = self.early_stopper(
                    val_dict["loss"], val_dict["dice_fullres"]
                )

                metric = (
                    self.early_stopper.best_dice
                    if self.early_stopper.criterion != "loss"
                    else self.early_stopper.best_loss
                )

                if improved:
                    model_save_path = self.rm.save_model(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.lr_scheduler,
                        epoch=epoch,
                        metric=metric,
                    )
                    self.logger.info(
                        f"New best model checkpoint saved to {model_save_path}"
                    )
                if stop:
                    break

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes.")

    def train_one_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        self.current_epoch = epoch
        self.mc.reset()

        for images, masks in tqdm(
            self.train_dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False
        ):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self._compute_loss(outputs, masks)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(
                    self.optimizer
                )  # Unscale gradients before clipping
                clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_norm
                )  # Or whatever parameters you clip
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
            batch_metrics = self._compute_metrics(outputs, masks)
            batch_metrics["loss"] = loss.item()
            self.mc.update(batch_metrics)
        return self.mc.aggregate()

    # def train_one_epoch(self, epoch: int) -> Dict[str, Any]:
    #     self.model.train()
    #     self.current_epoch = epoch
    #     self.mc.reset()

    #     for images, masks in tqdm(
    #         self.train_dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False
    #     ):
    #         images, masks = images.to(self.device), masks.to(self.device)
    #         self.optimizer.zero_grad()
    #         outputs = self.model(images)
    #         loss = self._compute_loss(outputs, masks)
    #         loss.backward()
    #         clip_grad_norm_(
    #             self.model.parameters(),
    #             max_norm=self.max_norm,
    #         )
    #         self.optimizer.step()
    #         batch_metrics = self._compute_metrics(outputs, masks)
    #         batch_metrics["loss"] = loss.item()
    #         self.mc.update(batch_metrics)
    #     return self.mc.aggregate()

    def validate(self, epoch: int) -> Dict[str, Any]:
        self.model.eval()
        self.mc.reset()
        with torch.no_grad():
            for images, masks in tqdm(
                self.val_dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False
            ):
                images, masks = images.to(self.device), masks.to(self.device)

                with autocast(self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, masks)

                batch_metrics = self._compute_metrics(outputs, masks)
                batch_metrics["loss"] = loss.item()
                self.mc.update(batch_metrics)
        return self.mc.aggregate()

    def _compute_loss(self, outputs, masks) -> torch.Tensor:
        if isinstance(outputs, (list, tuple)):
            raise TypeError(
                "Backbone-UNet: Output should be a single tensor, not a list or tuple."
            )
        return self.criterion(outputs, masks)

    def log_to_wandb(
        self, metrics: Dict[str, Any], step: int, prefix: str, commit: bool
    ):
        if self.wandb_logger is None or not self.wandb_logger.is_active:
            return

        log_dict: Dict[str, Any] = {"epoch": step}

        for key, value in metrics.items():
            full_key = f"{prefix}/{key}"

            if isinstance(value, (float, int)):
                log_dict[full_key] = value
            elif isinstance(value, list) and all(
                isinstance(v, (float, int)) for v in value
            ):
                # We always assume that a list of metric means is per-class
                log_dict[full_key] = {f"class_{i}": v for i, v in enumerate(value)}

        self.wandb_logger.log_metrics(log_dict, step=step, commit=commit)

    def _compute_metrics(
        self, outputs: torch.Tensor, masks: torch.Tensor
    ) -> Dict[str, Any]:
        if isinstance(outputs, (list, tuple)):
            raise TypeError(
                "Backbone-UNet: Output should be a single tensor, not a list or tuple."
            )
        preds = torch.argmax(outputs, dim=1)

        dice_scores_cls = metrics.dice_coefficient_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        iou_scores_cls = metrics.iou_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        precision_scores_cls = metrics.precision_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        recall_scores_cls = metrics.recall_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        f1_scores_cls = metrics.f1_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )

        return {
            "dice_fullres": safe_mean(dice_scores_cls),
            "iou_fullres": safe_mean(iou_scores_cls),
            "precision_fullres": safe_mean(precision_scores_cls),
            "recall_fullres": safe_mean(recall_scores_cls),
            "f1_fullres": safe_mean(f1_scores_cls),
            "dice_per_class_fullres": [x.item() for x in dice_scores_cls],
            "iou_per_class_fullres": [x.item() for x in iou_scores_cls],
            "precision_per_class_fullres": [x.item() for x in precision_scores_cls],
            "recall_per_class_fullres": [x.item() for x in recall_scores_cls],
            "f1_per_class_fullres": [x.item() for x in f1_scores_cls],
        }
