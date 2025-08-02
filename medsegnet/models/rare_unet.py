import logging
import numpy as np
from models.unet import UNet3D, _get_norm_layer, _get_activation_layer
import torch
from torch import batch_norm, nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

    
class DeeperResidualMSGate(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, activation_type, n_blocks=2):
        super().__init__()
        print("\n\n\n\nDeeperResidualMSGate: n_blocks=", n_blocks)
        # Identity mapper for the main skip connection
        self.identity_mapper = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Build the main path with multiple residual blocks
        layers = []
        current_channels = in_channels
        for i in range(n_blocks):
            layers.append(self.make_res_block(current_channels, out_channels, norm_type, activation_type))
            current_channels = out_channels # Subsequent blocks have same in/out channels
        
        self.res_path = nn.Sequential(*layers)
        self.final_activation = _get_activation_layer(activation_type)

    def make_res_block(self, in_c, out_c, norm_t, act_t):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            _get_norm_layer(norm_t, num_features=out_c),
            _get_activation_layer(act_t),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            _get_norm_layer(norm_t, num_features=out_c)
        )

    def forward(self, x):
        identity = self.identity_mapper(x)
        residual = self.res_path(x)
        
        out = identity + residual
        out = self.final_activation(out)
        return out

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
    

class DeeperStudyMSGate(nn.Module):
    """
    Deeper MSGate with multiple blocks for each gate.
    This is an ablation study version that allows for more complex gating mechanisms.
    """
    def __init__(self, in_channels, out_channels, n_gate_blocks, norm_type: str, activation_type: str, kernel_size=3):
        super(DeeperStudyMSGate, self).__init__()
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


class RAREUNet(UNet3D):
    """
    RARE-UNet = Resolution-Aligned Routing Entry U-Net
    """
    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)
        print(f"MSUNet3D: MODE={mode}")
        self.mode = mode
        self.n_ms_levels = int(
            cfg.architecture.get("num_multiscale_levels", self.depth - 1)
        )
        assert 0 < self.n_ms_levels < self.depth, "0 < self.n_ms_levels < self.depth"


        self.n_gate_blocks = cfg.architecture.get("n_gate_blocks", 2)

        # --- Learnable weights for segmentation and consistency losses (Doesn't work well rn)---
        # instead of fixed lists, we register two learnable vectors of size (n+1) and (n,)
        # w_logit_seg will be softmaxed to produce positive weights summing to 1 over [full, ms1,…]
        # n = self.n_ms_levels
        
        #classic problem in multi-task or multi-loss training: 
        # when we let the model learn the loss weights, it will often take the path of least resistance.
        # # w_logit_cons softmaxed over just the MS scales
        # self.w_logit_cons = nn.Parameter(torch.zeros(n))    
        
        #  # optional: initialize a bit of bias toward full-res
        #  #This encourages the model to first stabilize the main path, 
        #  # which serves as the "teacher" for the others.
        # self.w_logit_seg  = nn.Parameter(torch.zeros(n+1))  
        # with torch.no_grad():
        #     self.w_logit_seg[0] = 1.0 
        
        self.msb_blocks = nn.ModuleList()
        self.ms_heads = nn.ModuleList()
        

        # for each multiscale level 1...n_ms_levels
        for scale in range(1, self.n_ms_levels + 1):
            out_channels = min(self.n_filters * (2**scale), 320)
            
            self.msb_blocks.append(
                DeeperStudyMSGate(
                    self.in_channels,
                    out_channels,
                    norm_type=self.norm_type, 
                    activation_type=self.activation_type,
                    n_gate_blocks=self.n_gate_blocks
                )
            )
            in_channels = out_channels
            self.ms_heads.append(
                nn.Conv3d(in_channels, self.num_classes, kernel_size=1)
            )
            
        # total_scales = self.n_ms_levels + 1
        # self.ds_heads = nn.ModuleList()
        # for scale in range(total_scales):
        #     heads_at_scale = nn.ModuleList()
        #     for d in range(scale + 1, total_scales):
        #         in_channels = min(self.n_filters * (2**d), 320)
        #         heads_at_scale.append(
        #             nn.Conv3d(
        #                 in_channels,
        #                 self.num_classes,
        #                 kernel_size=1,
        #             )
        #         )
        #     self.ds_heads.append(heads_at_scale)
        
    
    def forward(self, x):
        if self.mode in ["inference", "test"]:
            return self.run_inference(x)
        else:
            return self.forward_train(x)

    def forward_train(self, x):
        
        # ===== Full resolution input =====
        full_seg = super().forward(x)
        
        # ===== Multiscale inputs (during training) =====
        ms_outputs     = []
        dec_feats_ms   = [] 
        D, H, W        = x.shape[2:]
        self.msb_feats = []  # msb1, msb2, msb3

        
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
            self.msb_feats.append(out_ms)
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

            dec_feats = []
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
                dec_feats.append(out_ms)

            # segmentation head for this multiscale level
            ms_seg = self.ms_heads[d - 1](out_ms)
            ms_outputs.append(ms_seg)
            
            # Store decoder features for this multiscale level
            dec_feats_ms.append(dec_feats)

        segmentations = (full_seg, *ms_outputs)
        consistency_pairs = tuple(zip(self.msb_feats, self.enc_feats_copy[1:]))

        return segmentations, consistency_pairs


        # # ===== Deep Supervision ===== 
        # total_scales = self.n_ms_levels + 1
        # dec_feats_full = self.dec_feats_copy[::-1]
        # dec_feats_ms   = [lst[::-1] for lst in dec_feats_ms]

        # ds_out = []
        # for scale in range(total_scales):
        #     heads = self.ds_heads[scale]
        #     this_scale_ds = []
        #     #0 + 1 + 0=1, ..., 0 + 1 + 
        #     # for each head supervising from scale k → d in (k+1…end)
        #     for head_idx, head in enumerate(heads): #type: ignore
        #         d = scale + 1 + head_idx   # the absolute decoder-scale you want
        #         if scale == 0:
        #             feat = dec_feats_full[d]
        #         else:
        #             feat = dec_feats_ms[scale-1][d - 1]
        #         this_scale_ds.append(head(feat))

        #     ds_out.append(this_scale_ds)
            
        # return segmentations, ds_out, consistency_pairs

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


class AlternativeMSUNet3D(RAREUNet):
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
