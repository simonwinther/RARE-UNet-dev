# transforms/base.py
import torch
from typing import Optional, Tuple

class Transform:
    """Base class for all image/mask transforms."""
    def __init__(self):
        pass  # reserved for future shared logic

    def __call__(
        self,
        image: torch.Tensor,             # C×D×H×W or 1×D×H×W
        mask:  Optional[torch.Tensor]=None  # 1×D×H×W
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError("Transform must implement __call__")

class RandomTransform(Transform):
    """Base class for random transforms applied with probability p."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p <= 1.0, "p must be in [0,1]"
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
        mask:  Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if torch.rand(()) < self.p:
            return self.apply(image, mask)
        # return unmodified, but keep mask's none-ness
        return image, mask

    def apply(
        self,
        image: torch.Tensor,
        mask:  Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform the actual transform. Must be overridden."""
        raise NotImplementedError("RandomTransform subclasses must implement apply()")
