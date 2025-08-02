import logging
from medsegnet.models.unet import UNet3D 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio


class UNetAugmented(UNet3D):
    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)
        print(f"BackboneAugmented: MODE={mode}")
        self.mode = mode
        self.target_shape_w_h_d = tuple(cfg.dataset.target_shape)
        W0, H0, D0 = self.target_shape_w_h_d

        self.crop_or_pad = tio.CropOrPad((W0, H0, D0))
        self.num_multiscale_levels = cfg.architecture.get(
            "num_multiscale_levels", self.depth - 1
        )
        self.interp_mode = "trilinear"
        self.align_corners = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (C, W, H, D)
        Applies 50% chance of multiscale augmentation then forwards.
        """

        prob_aug = torch.rand((), device=x.device).item()
        if prob_aug < 0.5 or self.mode in ["inference", "test"]:
            return super().forward(x)

        _, _, W0, H0, D0 = x.shape

        level = torch.randint(1, self.num_multiscale_levels + 1, (1,)).item()
        factor = 1.0 / (2**level)

        W1 = int(W0 * factor)
        H1 = int(H0 * factor)
        D1 = int(D0 * factor)

        if min(W1, H1, D1) < 1:
            raise ValueError(
                f"Scale level {level} is too aggressive: resulting size "
                f"({W1}, {H1}, {D1}) below 1 voxel"
            )

        x_bcwhd = x

        # downsample
        x_down = F.interpolate(
            x_bcwhd,
            size=(W1, H1, D1),
            mode=self.interp_mode,
            align_corners=self.align_corners,
        )

        if torch.rand(1).item() < 0.5:
            x_up = F.interpolate(
                x_down,
                size=(W0, H0, D0),
                mode=self.interp_mode,
                align_corners=self.align_corners,
            )
            x_bcwhd = x_up

        else:
            padded = []
            for sample in x_down:  # sample: (C, W1, H1, D1)
                arr = sample.detach().cpu().numpy()
                padded_arr = self.crop_or_pad(arr)
                vol = torch.from_numpy(padded_arr)
                vol = vol.to(device=x.device, dtype=x.dtype)
                padded.append(vol)
            # 4) stack back into shape (B, C, W0, H0, D0)
            x_bcwhd = torch.stack(padded, dim=0)

        return super().forward(x_bcwhd)
