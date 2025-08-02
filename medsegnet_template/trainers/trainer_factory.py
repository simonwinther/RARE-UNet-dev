from .trainer import BaseTrainer
from .trainerDS import DeepSupervisionTrainer
from .trainerMS import MultiscaleTrainer


def get_trainer(cfg, *args, **kwargs):
    arch = cfg.architecture.name
    if arch == "ds-unet3d":
        return DeepSupervisionTrainer(cfg, *args, **kwargs)
    elif arch == "ms-unet3d":
        return MultiscaleTrainer(cfg, *args, **kwargs)
    else:
        return BaseTrainer(cfg, *args, **kwargs)
