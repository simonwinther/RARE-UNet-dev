# transforms/intensity/random_noise.py
import torch

from transforms.base import RandomTransform

class RandomNoise(RandomTransform):
    """
    Add Gaussian noise to the image (only image).
    """
    def __init__(self, mean: float = 0.0, std: float = 0.01, p: float = 0.5):
        super().__init__(p=p)
        self.mean = mean
        self.std  = std

    def apply(self, image, mask=None):
        noise = torch.randn_like(image) * self.std + self.mean
        return image + noise, mask
