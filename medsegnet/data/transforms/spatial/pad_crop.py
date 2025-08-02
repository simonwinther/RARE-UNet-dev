"""
transforms/spatial/pad_crop.py

Contains two transforms:
- PadCrop: pad then center-crop to a target shape
- RandomPadCrop: pad then random-crop to a target shape (inherits RandomTransform)
"""

import random
from typing import Sequence, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transforms.base import Transform, RandomTransform


class PadCrop(Transform):
    """
    Pad the image (and optional mask) to at least target_shape, then center-crop to target_shape.
    """
    def __init__(
        self,
        target_shape: Sequence[int],
        pad_mode: str = 'constant',
        pad_value: Union[int, float] = 0
    ) -> None:
        super().__init__()
        self.target_shape: Tuple[int, ...] = tuple(target_shape)
        self.pad_mode: str = pad_mode
        self.pad_value: Union[int, float] = pad_value

    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Determine spatial dims
        spatial_ndim = len(self.target_shape)
        current = image.shape[-spatial_ndim:]

        # Compute padding
        pad: list[int] = []
        for curr, tgt in zip(reversed(current), reversed(self.target_shape)):
            diff = max(tgt - curr, 0)
            before = diff // 2
            after = diff - before
            pad.extend([before, after])

        # Apply padding
        if any(pad):
            image = F.pad(image, pad, mode=self.pad_mode, value=self.pad_value)
            if mask is not None:
                mask = F.pad(mask, pad, mode=self.pad_mode, value=self.pad_value)

        # Center crop
        new_shape = image.shape[-spatial_ndim:]
        starts = [(ns - tgt) // 2 for ns, tgt in zip(new_shape, self.target_shape)]
        slices = [slice(None)] * (image.dim() - spatial_ndim) + [
            slice(s, s + t) for s, t in zip(starts, self.target_shape)
        ]
        image = image[tuple(slices)]
        if mask is not None:
            mask = mask[tuple(slices)]

        return image, mask


class RandomPadCrop(RandomTransform):
    """
    Pad the image (and optional mask) to at least target_shape, then randomly crop.
    Inherits RandomTransform to handle probability.
    """
    def __init__(
        self,
        target_shape: Sequence[int],
        pad_mode: str = 'constant',
        pad_value: Union[int, float] = 0,
        p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.target_shape: Tuple[int, ...] = tuple(target_shape)
        self.pad_mode: str = pad_mode
        self.pad_value: Union[int, float] = pad_value

    def apply(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Determine spatial dims
        spatial_ndim = len(self.target_shape)
        current = image.shape[-spatial_ndim:]

        # Compute padding
        pad: list[int] = []
        for curr, tgt in zip(reversed(current), reversed(self.target_shape)):
            diff = max(tgt - curr, 0)
            before = diff // 2
            after = diff - before
            pad.extend([before, after])

        # Apply padding
        if any(pad):
            image = F.pad(image, pad, mode=self.pad_mode, value=self.pad_value)
            if mask is not None:
                mask = F.pad(mask, pad, mode=self.pad_mode, value=self.pad_value)

        # Random crop
        new_shape = image.shape[-spatial_ndim:]
        starts = [random.randint(0, ns - tgt) for ns, tgt in zip(new_shape, self.target_shape)]
        slices = [slice(None)] * (image.dim() - spatial_ndim) + [
            slice(s, s + t) for s, t in zip(starts, self.target_shape)
        ]
        image = image[tuple(slices)]
        if mask is not None:
            mask = mask[tuple(slices)]

        return image, mask
