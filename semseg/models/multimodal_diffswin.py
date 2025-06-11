import torch
import torch.nn as nn
import torch.nn.functional as F
from semseg.models.heads import SegFormerHead
from semseg.models.backbones.diffswin import DiffSwinTransformer
from semseg.models.modules.ffm import FeatureRectifyModule, FeatureFusionModule
from semseg.models.modules.mspa import MSPABlock
import os


class MultiModalDiffSwin(nn.Module):
    def __init__(self, num_classes=25, modals=['image', 'depth', 'event', 'lidar']):
        """
        MultimodalDiffSwin Model Initialization
        
        Initializes a multimodal segmentation model with separate DiffSwinTransformer encoders for each input modality.
        Handles feature rectification, fusion, and segmentation head construction.
        
        Args:
            num_classes (int): Number of output segmentation classes. Default: 25.
            modals (list[str]): List of input modalities (e.g. ['image', 'depth', 'event', 'lidar']). 
        
        Attributes:
            backbones (nn.ModuleDict): Independent DiffSwin encoders for each modality.
            frms (nn.ModuleDict): Feature rectification modules for non-image modalities.
            ffms (nn.ModuleList): Multi-scale feature fusion modules.
            self_query (MSPABlock): Cross-modal self-attention block.
            decode_head (SegFormerHead): Segmentation prediction head.
        """
        
        super().__init__()
        self.modals = modals
        self.num_modals = len(modals)

        # 为每个模态定义独立的 DiffSwin 编码器（不共享参数）
        self.backbones = nn.ModuleDict({
            modal: DiffSwinTransformer(
                pretrain_img_size=224,
                in_channels=3,
                embed_dims=128,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=(2, 2, 18, 2),
                num_heads=(4, 8, 16, 32),
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
            ) for modal in modals
        })

        self.frms = nn.ModuleDict({
            modal: FeatureRectifyModule(dim=128) for modal in modals if modal != 'image'
        })

        self.ffms = nn.ModuleList([
            FeatureFusionModule(dim=256, num_heads=8),
            FeatureFusionModule(dim=512, num_heads=16),
            FeatureFusionModule(dim=1024, num_heads=32)
        ])

        self.self_query = MSPABlock(dim=128)

        self.decode_head = SegFormerHead(
            [128, 256, 512, 1024],
            512,
            num_classes
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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained weight not found at: {path}")
        ckpt = torch.load(path, map_location='cpu')
        print(f"Loaded checkpoint keys: {list(ckpt.keys())[:5]} ... total: {len(ckpt)}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        print(f"Loading weights into all modals from: {path}")
        for modal in self.modals:
            msg = self.backbones[modal].load_state_dict(ckpt, strict=False)
            missing = len(msg.missing_keys)
            unexpected = len(msg.unexpected_keys)
            print(f"[Pretrained] {modal} - missing: {missing}, unexpected: {unexpected}")


    def forward(self, x_list):
        assert len(x_list) == self.num_modals, f"Expected {self.num_modals} inputs, got {len(x_list)}"
        feats_per_modal = {}
        for i, modal in enumerate(self.modals):
            x = x_list[i]
            t = torch.ones(x.shape[0], device=x.device)
            feats = self.backbones[modal](x, t)
            feats_per_modal[modal] = feats

        # stage 0: FRM + MSPA
        fused_stage0 = feats_per_modal['image'][0].clone()
        for modal in self.modals:
            if modal == 'image':
                continue
            aligned_img, aligned_other = self.frms[modal](feats_per_modal['image'][0], feats_per_modal[modal][0])
            fused_stage0 = torch.add(fused_stage0, aligned_other)

        fused_stage0 = fused_stage0 / self.num_modals
        fused_stage0 = self.self_query(fused_stage0)

        # stage 1-3: FFM融合
        fused_feats = [fused_stage0]
        for i in range(1, 4):
            feat_list = [feats_per_modal[modal][i] for modal in self.modals]
            fused = self.ffms[i - 1](feat_list[0], feat_list[1])
            for j in range(2, len(feat_list)):
                fused = self.ffms[i - 1](fused, feat_list[j])
            fused_feats.append(fused)

        out = self.decode_head(fused_feats)
        out = F.interpolate(out, size=x_list[0].shape[2:], mode='bilinear', align_corners=False)
        return out

if __name__ == '__main__':
    pretrained_path = "/home/chagall/research/Weights/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth"
    model = MultiModalDiffSwin(num_classes=25)
    model.init_pretrained(pretrained_path)
    inputs = [
        torch.randn(1, 3, 512, 512),
        torch.randn(1, 3, 512, 512),
        torch.randn(1, 3, 512, 512),
        torch.randn(1, 3, 512, 512),
    ]
    output = model(inputs)
    print(output.shape)
