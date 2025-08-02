# transforms/compose.py
from typing import List, Optional, Tuple
import torch

from transforms.base import Transform

class Compose(Transform):
    """Chains together a list of transforms."""
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
