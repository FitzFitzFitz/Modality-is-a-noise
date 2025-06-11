from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from semseg.models.heads import SegFormerHead
from semseg.models.backbones.diffswin import DiffSwinTransformer
from semseg.models.modules.ffm import FeatureRectifyModule, FeatureFusionModule
from semseg.models.modules.mspa import MSPABlock


class PredictorLG(nn.Module):
    """Soft token selector for each extra modality (supports 4‑D or token input)."""
    def __init__(self, embed_dim: int, num_modals: int):
        super().__init__()
        self.score_nets = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid(),
            ) for _ in range(num_modals)
        ])

    def forward(self, x_list, t=None, return_feats=False):
        B = x_list[0].size(0)
        if t is None:
            t = torch.randint(1, self.max_t + 1, (B,), device=x_list[0].device, dtype=torch.float)

        feats_per_modal = {m: self.backbone(x, t) for x, m in zip(x_list, self.modals)}
        fused_feats, dn_accum = [], 0.0

        for i in range(4):
            feat_img = feats_per_modal['image'][i]
            others = [feats_per_modal[m][i] for m in self.modals if m != 'image']
            weights = self.predictors[i](others)
            others = [o * w + o for o, w in zip(others, weights)]
            feat_mod = torch.max(torch.stack(others, 0), 0).values

            feat_img, feat_mod = self.frms[i](feat_img, feat_mod)
            fused = self.ffms[i](feat_img, feat_mod)
            fused = self.mspa_blocks[i](fused)
            fused_feats.append(fused)
            dn_accum = F.l1_loss(feat_mod, feat_img.detach()) + dn_accum

        seg = self.decode_head(fused_feats)
        seg = F.interpolate(seg, size=x_list[0].shape[2:], mode='bilinear', align_corners=False)
        dn_loss = dn_accum / 4.0

        # ---- output protocol ----
        # * eval/inference: logits only (for .softmax())
        # * training: (logits, aux)
        # * analysis: optionally return feats too
        if return_feats:
            return seg, fused_feats, dn_loss
        if self.training:
            return seg, dn_loss
        return seg





class DiffModalSegV4(nn.Module):
    """Diffusion‑inspired multi‑modal segmentation – CMNeXt‑scale."""

    BACKBONE_SETTINGS = {
        'tiny': dict(embed_dims=96,  depths=(2, 2,  6, 2), num_heads=(3,  6, 12, 24)),
        'base': dict(embed_dims=128, depths=(2, 2, 18, 2), num_heads=(4,  8, 16, 32)),
    }

    def __init__(self, backbone: str = 'v1-base', num_classes: int = 25,
                 modals: list[str] | None = None, max_t: int = 1000):
        super().__init__()
        self.modals = modals or ['image', 'depth', 'event', 'nir']
        assert 'image' in self.modals, "Need reference 'image' modality"
        self.max_t = max_t

        size_key = backbone.split('-')[-1]
        cfg = self.BACKBONE_SETTINGS[size_key]
        C = cfg['embed_dims']

        # shared DiffSwin backbone
        self.backbone = DiffSwinTransformer(
            pretrain_img_size=224,
            in_channels=3,
            embed_dims=C,
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=cfg['depths'],
            num_heads=cfg['num_heads'],
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            drop_path_rate=0.1,
            learned_sinusoidal_dim=16,
            norm_cfg=dict(type='LN'))

        # stage modules
        self.frms = nn.ModuleList([FeatureRectifyModule(dim=C * (2 ** i)) for i in range(4)])
        self.predictors = nn.ModuleList([
            PredictorLG(C * (2 ** i), num_modals=len(self.modals) - 1)
            for i in range(4)])
        self.mspa_blocks = nn.ModuleList([MSPABlock(C * (2 ** i)) for i in range(4)])
        num_heads_stage = [1, 3, 6, 12]  # divisible factors
        self.ffms = nn.ModuleList([
            FeatureFusionModule(C * (2 ** i), num_heads=num_heads_stage[i])
            for i in range(4)])

        self.decode_head = SegFormerHead([C, 2*C, 4*C, 8*C], 512, num_classes)
        self._init_weights()

    # ---------------------------------------------------------------
    def init_pretrained(self, path: str | None):
        if not path:
            return
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict):
            ckpt = ckpt.get('state_dict', ckpt.get('model', ckpt))
        self.backbone.load_state_dict(ckpt, strict=False)

    # ---------------------------------------------------------------
    def forward(self, x_list, t=None, return_feats=False):
        B = x_list[0].size(0)
        if t is None:
            t = torch.randint(1, self.max_t + 1, (B,), device=x_list[0].device, dtype=torch.float)

        feats_per_modal = {m: self.backbone(x, t) for x, m in zip(x_list, self.modals)}
        fused_feats, dn_accum = [], 0.0

        for i in range(4):
            feat_img = feats_per_modal['image'][i]
            others = [feats_per_modal[m][i] for m in self.modals if m != 'image']
            weights = self.predictors[i](others)
            others = [o * w + o for o, w in zip(others, weights)]
            feat_mod = torch.max(torch.stack(others, 0), 0).values

            feat_img, feat_mod = self.frms[i](feat_img, feat_mod)
            fused = self.ffms[i](feat_img, feat_mod)
            fused = self.mspa_blocks[i](fused)
            fused_feats.append(fused)

            # accumulate denoise loss (detach to stop img grad)
            dn_accum = F.l1_loss(feat_mod, feat_img.detach()) + dn_accum

        seg = self.decode_head(fused_feats)
        seg = F.interpolate(seg, size=x_list[0].shape[2:], mode='bilinear', align_corners=False)
        dn_loss = dn_accum / 4.0

        if return_feats:
            return seg, fused_feats, dn_loss
        return seg, dn_loss

    # ---------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-2, b=2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
