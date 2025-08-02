# config/schema.py

# Potential Idea for Better Type Safety in Code As Well! :D 
# I might implement, but its weird with hydra, so see if I 
# Can find a good solution or maybe switch
# to argparser and use this as "default" config instead
#of hydra default config, because
#thats why its weird having hydra and this, because
#theyre both "default" configs

from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict

# ===================================================================
# Nested Schemas (for organization)
# ===================================================================

@dataclass
class GPUSchema:
    mode: str = "single"
    devices: List[int] = field(default_factory=lambda: [0])
    backend: str = "nccl"

@dataclass
class FileLoggingSchema:
    level: str = "debug"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

@dataclass
class ConsoleLoggingSchema:
    level: str = "info"
    format: str = "[%(levelname)s]: %(message)s"
    datefmt: str = "%H:%M:%S"

@dataclass
class LoggingSchema:
    disable_all: bool = False
    file: FileLoggingSchema = field(default_factory=FileLoggingSchema)
    console: ConsoleLoggingSchema = field(default_factory=ConsoleLoggingSchema)
    ppmat: bool = False

@dataclass
class OptimizationSchema:
    mode: str = "deterministic"

@dataclass
class WandbSchema:
    log: bool = False
    wandb_project: str = "MedicalSegmentation"
    name: str = "deleteme"
    group: str = "${dataset.name}"
    tags: List[str] = field(default_factory=lambda: ["${dataset.name}"])

@dataclass
class TrainedModelsSchema:
    base_dir: str = "trained_models"

@dataclass
class SummarySchema:
    verbose: int = 0
    col_names: List[str] = field(default_factory=lambda: ["input_size", "output_size", "num_params"])

# This schema is comprehensive to cover all architecture YAMLs.
# Optional fields will be `None` if not specified in the active YAML.
@dataclass
class ArchitectureSchema:
    _target_: str = "models.UNet.UNet3D"
    summary: SummarySchema = field(default_factory=SummarySchema)
    name: str = "unet3d"
    pp: str = "unet3d"
    depth: int = 5
    n_filters: int = 32
    dropout: float = 0.2
    drop_last: bool = True
    norm_type: str = "instance"
    activation_type: str = "leaky_relu"
    
    # RARE-UNet specific fields
    cons_loss_start_epoch: Optional[int] = None
    num_multiscale_levels: Optional[int] = None

    # DS-UNet 
    ds_levels: Optional[int] = None
    fusion: Optional[str] = None
    
    # UNet-Aug specific fields (No new fields, just a different target)

@dataclass
class DataAugmentationSchema:
    enabled: bool = False
    transforms: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LossSchema:
    name: str = "CombinedLoss"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "_target_": "utils.losses.CombinedLoss", "alpha": 0.35, "ignore_index": 0
    })

@dataclass
class OptimizerSchema:
    name: str = "AdamW"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "_target_": "torch.optim.AdamW", "lr": 3e-4, "weight_decay": 3e-5
    })

@dataclass
class SchedulerSchema:
    name: str = "ReduceLROnPlateau"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "mode": "max", "factor": 0.5, "patience": 10, "threshold": 0.001,
        "cooldown": 0, "min_lr": 1e-6
    })

@dataclass
class EarlyStopperSchema:
    patience: int = 35
    verbose: bool = True
    delta: float = 1e-3
    criterion: str = "dice"

@dataclass
class TrainingSchema:
    num_epochs: int = 80
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    grad_clip_norm: float = 1.0
    weight_decay: float = 3e-5
    learning_rate: float = 3e-4
    data_augmentation: DataAugmentationSchema = field(default_factory=DataAugmentationSchema)
    loss: LossSchema = field(default_factory=LossSchema)
    optimizer: OptimizerSchema = field(default_factory=OptimizerSchema)
    scheduler: SchedulerSchema = field(default_factory=SchedulerSchema)
    early_stopper: EarlyStopperSchema = field(default_factory=EarlyStopperSchema)

# This schema is comprehensive for all dataset YAMLs.
@dataclass
class DatasetSchema:
    name: str = "Task04_Hippocampus"
    target_shape: List[int] = field(default_factory=lambda: [32, 64, 32])
    base_path: str = "../datasets/Task04_Hippocampus/"
    images_subdir: str = "imagesTr"
    labels_subdir: str = "labelsTr"
    num_classes: int = 3
    in_channels: int = 1

# ===================================================================
# The Main, Unified Application Config
# ===================================================================

@dataclass
class AppConfig:
    # This defaults list is a placeholder. Hydra will build the final config
    # based on the YAML files you use, and then load it into this schema.
    defaults: List[Any] = field(default_factory=lambda: [
        "_self_",
        {"architecture": "unet3d"}, # Example default
        {"dataset": "Task04_Hippocampus"}, # Example default
        {"training": "default"},
    ])

    # --- Config groups ---
    architecture: ArchitectureSchema = field(default_factory=ArchitectureSchema)
    dataset: DatasetSchema = field(default_factory=DatasetSchema)
    training: TrainingSchema = field(default_factory=TrainingSchema)
    
    # --- Top-level settings from base.yaml ---
    gpu: GPUSchema = field(default_factory=GPUSchema)
    logging: LoggingSchema = field(default_factory=LoggingSchema)
    optimization: OptimizationSchema = field(default_factory=OptimizationSchema)
    wandb: WandbSchema = field(default_factory=WandbSchema)
    trained_models: TrainedModelsSchema = field(default_factory=TrainedModelsSchema)

    skip_metric_computation: bool = True
    use_amp: bool = True
    seed: int = 42
    
    # --- Optional top-level settings ---
    dataset_subset_size: Optional[int] = None
    resume_checkpoint: Optional[str] = "im a nigger"