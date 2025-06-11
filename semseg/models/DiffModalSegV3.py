# semseg/models/DiffModalSegV3.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from semseg.models.heads import SegFormerHead
from semseg.models.backbones.diffswin import DiffSwinTransformer
from semseg.models.modules.ffm import FeatureRectifyModule, FeatureFusionModule
from semseg.models.modules.mspa import MSPABlock

# ------------------------------------------------------------
#   Utility: soft token selector (PredictorLG variant)
# ------------------------------------------------------------
class PredictorLG(nn.Module):
    """给每个额外模态学习一张 soft‑selection 权重图"""
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

    def forward(self, feats):
        outs = []
        for net, feat in zip(self.score_nets, feats):
            if feat.dim() == 4:                  # B C H W → B 1 H W
                B, C, H, W = feat.shape
                tokens = feat.permute(0, 2, 3, 1).reshape(B, -1, C)
                score = net(tokens).view(B, 1, H, W)
            else:                                # B N C → B 1 N
                score = net(feat).transpose(1, 2)
            outs.append(score)
        return outs


# ------------------------------------------------------------
#   Main model
# ------------------------------------------------------------
class DiffModalSegV3(nn.Module):
    """Diffusion‑inspired multimodal segmentation (≈CMNeXt 参数量)"""

    BACKBONE_SETTINGS = {
        'tiny': dict(embed_dims=96,  depths=(2, 2,  6, 2), num_heads=(3,  6, 12, 24)),
        'base': dict(embed_dims=128, depths=(2, 2, 18, 2), num_heads=(4,  8, 16, 32)),
    }

    def __init__(
        self,
        backbone: str = 'v3-tiny',
        num_classes: int = 25,
        modals: list[str] | None = None,
        max_t: int = 1000,
        dn_weight: float = 0,   # ← **可配置的 denoising 损失权重**
    ):
        super().__init__()
        self.modals = modals or ['image', 'depth', 'event', 'lidar']
        if self.modals[0] != 'image':
            raise ValueError("第一个模态必须是 'image' 作为主参考分支。")
        self.num_modals = len(self.modals)
        self.max_t     = max_t
        self.dn_weight = dn_weight

        size_key = backbone.split('-')[-1]
        if size_key not in self.BACKBONE_SETTINGS:
            raise KeyError(f'Unsupported backbone size: {size_key}')
        cfg = self.BACKBONE_SETTINGS[size_key]

        # -------- shared DiffSwin backbone (时间调制) --------
        self.backbone = DiffSwinTransformer(
            pretrain_img_size=224,
            in_channels=3,
            embed_dims=cfg['embed_dims'],
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=cfg['depths'],
            num_heads=cfg['num_heads'],
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            patch_norm=True,
            drop_path_rate=0.1,
            use_abs_pos_embed=False,
            learned_sinusoidal_dim=16,
            norm_cfg=dict(type='LN'),
        )

        C = cfg['embed_dims']                       # stage0 channel
        self.frms        = nn.ModuleList([FeatureRectifyModule(C * 2**i) for i in range(4)])
        self.predictors  = nn.ModuleList([PredictorLG(C * 2**i, self.num_modals - 1) for i in range(4)])
        self.mspa_blocks = nn.ModuleList([MSPABlock(C * 2**i) for i in range(4)])

        # FFM - 自适应确定可整除的 heads 数
        self.ffms = nn.ModuleList([
            FeatureFusionModule(
                dim      = C * 2**i,
                num_heads= max(1, (C * 2**i) // 64)  # 保证 dim % heads == 0
            ) for i in range(4)
        ])

        self.decode_head = SegFormerHead([C, C*2, C*4, C*8], 512, num_classes)

        # -------- init weights (递归 apply) --------
        self.apply(self._init_weights)

    # -------------------------------------------------------
    # Weight init  (static so apply(fn) 只需一个参数)
    # -------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # -------------------------------------------------------
    #  预训练 Swin 权重加载
    # -------------------------------------------------------
    def init_pretrained(self, path: str | None):
        if not path:
            print('[DiffModalSegV3] skip loading pretrained (empty path)'); return
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict):
            ckpt = ckpt.get('state_dict', ckpt.get('model', ckpt))
        msg = self.backbone.load_state_dict(ckpt, strict=False)
        print('[DiffModalSegV3] loaded Swin backbone:', msg)

    # -------------------------------------------------------
    #  Forward
    # -------------------------------------------------------
    def forward(self, xs, t=None, return_feats=False):
        """
        xs: List[Tensor]  按 self.modals 顺序 (B,3,H,W)
        return:
            训练   → (logits, aux_loss_tensor)
            推理   → logits
            return_feats=True → (logits, fused_feats, aux_loss)
        """
        if len(xs) != self.num_modals:
            raise ValueError(f'Expect {self.num_modals} inputs, got {len(xs)}')

        B, _, H, W = xs[0].shape
        if t is None:
            t = torch.randint(1, self.max_t+1, (B,), device=xs[0].device).float()

        # 1) backbone features
        feats = {m: self.backbone(x, t) for m, x in zip(self.modals, xs)}

        # 2) stage-wise fusion
        fused_feats = []
        for i in range(4):
            feat_img = feats['image'][i]                 # B C h w
            others   = [feats[m][i] for m in self.modals if m != 'image']

            # soft token selection
            weights  = self.predictors[i](others)
            others   = [f*w + f for f, w in zip(others, weights)]
            feat_mod = torch.max(torch.stack(others, 0), 0).values

            # FRM → FFM → MSPA
            feat_img, feat_mod = self.frms[i](feat_img, feat_mod)
            fused = self.ffms[i](feat_img, feat_mod)
            fused = self.mspa_blocks[i](fused)
            fused_feats.append(fused)

        # 3) decode
        logits = self.decode_head(fused_feats)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        # optional denoising consistency loss (scaled by self.dn_weight)
        dn_loss = torch.tensor(0., device=logits.device)
        if self.training and torch.is_grad_enabled() and self.dn_weight > 0:
            raw = sum(F.l1_loss(fm, fi.detach()) for fm, fi in zip(fused_feats, feats['image']))
            dn_loss = raw * self.dn_weight

        if return_feats:
            return logits, fused_feats, dn_loss
        return (logits, dn_loss) if self.training and torch.is_grad_enabled() else logits


# ---------------- quick sanity check ----------------
if __name__ == '__main__':
    model = DiffModalSegV3('v3-tiny', num_classes=19, modals=['image', 'depth'])
    x = [torch.randn(1,3,512,512) for _ in range(2)]
    out = model(x)
    if isinstance(out, tuple):
        print('logits shape:', out[0].shape, 'dn_loss:', out[1].item())
    else:
        print('logits shape:', out.shape)
