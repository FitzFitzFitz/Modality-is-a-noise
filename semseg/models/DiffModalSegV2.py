import torch
import torch.nn as nn
import torch.nn.functional as F
from semseg.models.heads import SegFormerHead
from semseg.models.backbones.diffswin import DiffSwinTransformer
from semseg.models.modules.ffm import FeatureRectifyModule, FeatureFusionModule
from semseg.models.modules.mspa import MSPABlock


class DiffModalSegV2(nn.Module):
    def __init__(self, backbone: str = 'v1-base', num_classes=25, modals=['image', 'depth', 'event', 'lidar']):
        super().__init__()
        self.modals = modals
        self.num_modals = len(modals)

        size_key = backbone.split('-')[-1]  # 支持 v1-base / v2-base 等

        backbone_settings = {
            'tiny': dict(embed_dims=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)),
            'base': dict(embed_dims=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32)),
        }
        assert size_key in backbone_settings, f"Unsupported backbone variant: {size_key}"
        settings = backbone_settings[size_key]

        # 共享 DiffSwinTransformer 编码器
        self.backbone = DiffSwinTransformer(
            pretrain_img_size=224,
            in_channels=3,
            embed_dims=settings['embed_dims'],
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=settings['depths'],
            num_heads=settings['num_heads'],
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            patch_norm=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            use_abs_pos_embed=False,
            learned_sinusoidal_dim=16,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            with_cp=False,
            pretrained=None,
            frozen_stages=-1,
            init_cfg=None
        )

        embed_dims = settings['embed_dims']
        self.frms = nn.ModuleDict({
            modal: FeatureRectifyModule(dim=embed_dims) for modal in modals if modal != 'image'
        })

        self.self_query = MSPABlock(dim=embed_dims)

        self.ffms = nn.ModuleList([
            FeatureFusionModule(dim=embed_dims * 2, num_heads=8),
            FeatureFusionModule(dim=embed_dims * 4, num_heads=16),
            FeatureFusionModule(dim=embed_dims * 8, num_heads=32)
        ])

        self.decode_head = SegFormerHead(
            [embed_dims, embed_dims * 2, embed_dims * 4, embed_dims * 8], 512, num_classes
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def init_pretrained(self, path):
        ckpt = torch.load(path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        msg = self.backbone.load_state_dict(ckpt, strict=False)
        print(msg)

    def forward(self, x_list):
        assert len(x_list) == self.num_modals
        B = x_list[0].shape[0]

        feats_per_modal = {}
        for x, modal in zip(x_list, self.modals):
            t = torch.ones(B, device=x.device)
            feats = self.backbone(x, t)
            feats_per_modal[modal] = feats

        # stage 0：FRM + MSPABlock
        feat_img = feats_per_modal['image'][0].clone()
        fused_stage0 = feat_img
        for modal in self.modals:
            if modal == 'image':
                continue
            _, feat_modal = self.frms[modal](feat_img, feats_per_modal[modal][0])
            fused_stage0 = fused_stage0 + feat_modal
        fused_stage0 = fused_stage0 / self.num_modals
        fused_stage0 = self.self_query(fused_stage0)

        # stage 1~3：FFM 逐级融合
        fused_feats = [fused_stage0]
        for i in range(1, 4):
            feat_list = [feats_per_modal[modal][i] for modal in self.modals]
            fused = feat_list[0]
            for j in range(1, len(feat_list)):
                fused = self.ffms[i - 1](fused, feat_list[j])
            fused_feats.append(fused)

        out = self.decode_head(fused_feats)
        out = F.interpolate(out, size=x_list[0].shape[2:], mode='bilinear', align_corners=False)
        return out