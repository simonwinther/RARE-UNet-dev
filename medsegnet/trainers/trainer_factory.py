# trainers/trainer_factory.py

from trainers.rare_trainer import RARETrainer
from trainers.trainer import BaseTrainer
from trainers.trainer_ds import DeepSupervisionTrainer

TRAINER_REGISTRY = {
    "rare_unet": RARETrainer,
    "unet": BaseTrainer,
    "unet_aug": BaseTrainer,
    "unet_ds": DeepSupervisionTrainer,
}

def get_trainer(cfg, *args, **kwargs):
    """
    Factory function to get a trainer instance based on the config.
    """
    arch = cfg.architecture.name
    trainer_class = TRAINER_REGISTRY.get(arch.lower())

    if trainer_class is None:
        available = ", ".join(TRAINER_REGISTRY.keys()) if TRAINER_REGISTRY else "None"
        raise ValueError(
            f"Trainer for architecture {arch!r} is not implemented.\n"
            f"Available trainers: [{available}]"
        )
        
    return trainer_class(cfg, *args, **kwargs)