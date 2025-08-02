import logging
from os import close
from platform import architecture
import numpy as np
from models.UNet import UNet3D, _get_norm_layer, _get_activation_layer
import torch
from torch import batch_norm, nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MSGate(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type: str, activation_type: str, kernel_size=3):
            super(MSGate, self).__init__()
            layers = [
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
                _get_norm_layer(norm_type, num_features=out_channels),
                _get_activation_layer(activation_type),
                nn.Conv3d(out_channels, out_channels, kernel_size, padding=1),
                _get_norm_layer(norm_type, num_features=out_channels),
                _get_activation_layer(activation_type),
            ]
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
    
class ResidualMSGate(nn.Module):
    """
    A more powerful MSGate using a residual connection.
    This helps stabilize training and allows the gate to learn a more
    complex transformation from low-res input to high-res feature space.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualMSGate, self).__init__()

        batch_norm = True  # delete this shit...

        # Use a 1x1 conv for the identity path to match the output channels.
        # This is the "projection shortcut" from the original ResNet paper.
        self.identity_mapper = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        # The main residual path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True) if batch_norm else nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True) if batch_norm else nn.Identity()

    def forward(self, x):
        # Map the input to the output dimension for the skip connection
        identity = self.identity_mapper(x)
        
        # Forward pass through the main path
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        # Add the skip connection (the "residual" part)
        out += identity
        out = self.relu(out)
        
        return out

class AblationStudyMSGate(nn.Module):
    #change to _getnormlayer and _get_activation_layer """
    #change to _getnormlayer and _get_activation_layer """
    #change to _getnormlayer and _get_activation_layer """
    #change to _getnormlayer and _get_activation_layer """
    def __init__(self, in_channels, out_channels, n_gate_blocks, norm_type: str, activation_type: str, kernel_size=3):
        super(AblationStudyMSGate, self).__init__()
        layers = []
        for idx in range(n_gate_blocks):
            in_channels = in_channels if idx == 0 else out_channels
            layers.extend([
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
                _get_norm_layer(norm_type, num_features=out_channels),
                _get_activation_layer(activation_type),
            ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


# class RareUNet(nn.Module):
#     """
#     RARE = Resolution-Aligned Routing Entry U-Net
#     """
class MSUNet3D(UNet3D):
    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)
        print(f"MSUNet3D: MODE={mode}")
        self.mode = mode
        self.n_ms_levels = int(
            cfg.architecture.get("num_multiscale_levels", self.depth - 1)
        )
        assert 0 < self.n_ms_levels < self.depth, "0 < self.n_ms_levels < self.depth"
        
        if cfg.architecture.get("ablation", None) is not None:
            logger.warning("You're currently using an ablation study version of MSUNet3D.")   
            self.n_gate_blocks = int(cfg.architecture.ablation.get("num_gate_blocks", 2))
            assert 0 < self.n_gate_blocks, "0 < self.n_gate_blocks"

        self.msb_blocks = nn.ModuleList()
        self.ms_heads = nn.ModuleList()
        for k in range(1, self.depth):
            out_channels = min(self.n_filters * (2**k), 320)
            self.msb_blocks.append(
                MSGate(
                    self.in_channels,
                    out_channels,
                    norm_type=self.norm_type, 
                    activation_type=self.activation_type,
                )
            )
            in_channels = out_channels
            self.ms_heads.append(
                nn.Conv3d(in_channels, self.num_classes, kernel_size=1)
            )
    
    def forward(self, x):
        if self.mode in ["inference", "test"]:
            return self.run_inference(x)
        else:
            return self.forward_train(x)

    def forward_train(self, x):
        full_seg = super().forward(x)
        # ===== Multiscale inputs (during training) =====
        ms_outputs = []
        D, H, W = x.shape[2:]
        msb_feats = []  # msb1, msb2, msb3
        for d in range(1, self.n_ms_levels + 1):
            # Downsampling
            target_size = (D // (2**d), H // (2**d), W // (2**d))
            x_ms = F.interpolate(
                x.detach(), size=target_size, mode="trilinear", align_corners=False
            )

            # Build encoder features for MS path
            ms_feats = []
            msb = self.msb_blocks[d - 1]
            out_ms = msb(x_ms)
            msb_feats.append(out_ms)
            ms_feats.append(out_ms)
            out_ms = self.pools[d](out_ms)
            out_ms = self.enc_dropouts[d](out_ms)

            for enc, pool, dropout in zip(
                list(self.encoders)[d + 1 :],
                list(self.pools)[d + 1 :],
                list(self.enc_dropouts)[d + 1 :],
            ):
                out_ms = enc(out_ms)
                ms_feats.append(out_ms)
                out_ms = dropout(pool(out_ms))

            # Bottleneck
            out_ms = self.bn(out_ms)

            num_ups = self.depth - d

            # Decoder up to match MS scale
            for up_conv, dec, drop in zip(
                list(self.up_convs)[:num_ups],
                list(self.decoders)[:num_ups],
                list(self.dec_dropouts)[:num_ups],
            ):
                out_ms = up_conv(out_ms)
                skip = ms_feats.pop()
                out_ms = torch.cat([out_ms, skip], dim=1)
                out_ms = dec(out_ms)
                out_ms = drop(out_ms)

            ms_seg = self.ms_heads[d - 1](out_ms)
            ms_outputs.append(ms_seg)

        segmentations = (full_seg, *ms_outputs)
        consistency_pairs = tuple(zip(msb_feats, self.enc_feats_copy[1:]))

        return segmentations, consistency_pairs

    def run_inference(self, x):
        W, H, D = x.shape[2:]
        input_shape = (W, H, D)

        def _div_shape(shape, factor):
            return tuple(s // factor for s in shape)

        target_shape = tuple(self.target_shape)
        depth = self.depth

        # build mapping shape -> entry string
        shape_to_entry = {target_shape: "enc1"}
        for d in range(1, self.n_ms_levels + 1):
            key = _div_shape(target_shape, 2**d)
            shape_to_entry[key] = f"msb{d}"

        allowed_shapes = list(shape_to_entry.keys())
        rounded = tuple(2 ** round(np.log2(s)) for s in input_shape)

        if rounded not in shape_to_entry:
            raise ValueError(
                f"Input shape {input_shape} is not in allowed shapes {allowed_shapes}"
            )

        # get entry point
        entry_gateway = shape_to_entry[rounded]

        if entry_gateway == "enc1":
            # full resolution
            out = x
            encoder_feats = []
            for enc, pool, drop in zip(self.encoders, self.pools, self.enc_dropouts):
                out = enc(out)
                encoder_feats.append(out)
                out = drop(pool(out))

            # bottleneck
            out = self.bn(out)

            # Decoder pathway
            for up_conv, decoder, drop in zip(
                self.up_convs, self.decoders, self.dec_dropouts
            ):
                out = up_conv(out)
                skip = encoder_feats.pop()
                out = torch.cat([out, skip], dim=1)
                out = decoder(out)
                out = drop(out)

            final_out = self.final_conv(out)
            return final_out
        elif entry_gateway.startswith("msb"):
            # lower resolution image
            level = int(entry_gateway.replace("msb", ""))
            msb = self.msb_blocks[level - 1]
            out = msb(x)
            ms_feats = []
            ms_feats.append(out)
            out = self.pools[level](out)
            out = self.enc_dropouts[level](out)

            for enc, pool, drop in zip(
                list(self.encoders)[level + 1 :],
                list(self.pools)[level + 1 :],
                list(self.enc_dropouts)[level + 1 :],
            ):
                out = enc(out)
                ms_feats.append(out)
                out = drop(pool(out))

            # bottleneck
            out = self.bn(out)

            num_ups = depth - level
            # decoder up to match MS scale
            for up_conv, dec, drop in zip(
                list(self.up_convs)[:num_ups],
                list(self.decoders)[:num_ups],
                list(self.dec_dropouts)[:num_ups],
            ):
                out = up_conv(out)
                skip = ms_feats.pop()
                out = torch.cat([out, skip], dim=1)
                out = dec(out)
                out = drop(out)

            final_out = self.ms_heads[level - 1](out)  # ms_heads not final_conv
            return final_out
        else:
            raise ValueError(f"Unknown entry point in Multiscale UNet: {entry_gateway}")


class AlternativeMSUNet3D(MSUNet3D):
    def forward(self, x):
        D, H, W = x.shape[2:]
        outputs = super().forward(x)
        ms_outputs = [
            ms_head(dec_feat)
            for ms_head, dec_feat in zip(self.ms_heads, self.dec_feats_copy[1:])
        ]
        msb_feats = []
        for d in range(1, self.depth):
            target_size = (D // (2**d), H // (2**d), W // (2**d))
            downsampled_x = F.interpolate(
                x, size=target_size, mode="trilinear", align_corners=False
            )
            msb = self.msb_blocks[d - 1]
            out_ms = msb(downsampled_x)
            msb_feats.append(out_ms)
        segmentations = (outputs, *ms_outputs)
        consistency_pairs = tuple(zip(msb_feats, self.enc_feats_copy[1:]))
        return segmentations, consistency_pairs
