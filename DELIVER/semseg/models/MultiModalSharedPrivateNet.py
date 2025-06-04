import torch
import torch.nn as nn
import torch.nn.functional as F
from semseg.models.heads import SegFormerHead
from semseg.models.backbones.diffswin import DiffSwinTransformer
from semseg.models.modules.mspa import MSPABlock


def infer_swin_config_from_checkpoint(path: str):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))

    embed_dim = None
    depths = [0, 0, 0, 0]
    num_heads = [0, 0, 0, 0]

    for k, v in state_dict.items():
        if embed_dim is None and "patch_embed.norm.weight" in k:
            embed_dim = v.shape[0]
        for i in range(4):
            if f"stages.{i}.blocks." in k:
                depths[i] += 1
            if f"stages.{i}.blocks.0.attn.relative_position_bias_table" in k:
                num_heads[i] = v.shape[1]

    if embed_dim is None or sum(depths) == 0:
        raise RuntimeError("❌ 无法从权重中解析 Swin 配置")

    return embed_dim, depths, num_heads, state_dict


class SharedPrivateEncoder(nn.Module):
    def __init__(self, variant='MMSP'):
        super().__init__()
        self.variant = variant
        self.backbone = None
        self.split = None
        self.out_dims = None

    def init_pretrained(self, path):
        print(f"[SharedPrivateEncoder] 加载预训练权重: {path}")
        embed_dim, depths, num_heads, state_dict = infer_swin_config_from_checkpoint(path)

        self.backbone = DiffSwinTransformer(
            pretrain_img_size=224,
            in_channels=3,
            embed_dims=embed_dim,
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=tuple(depths),
            num_heads=tuple(num_heads),
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

        self.out_dims = [embed_dim * (2 ** i) for i in range(4)]
        self.split = nn.ModuleList([
            nn.Conv2d(dim, dim * 2, kernel_size=1) for dim in self.out_dims
        ])

        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"✅ 构建完成: embed_dim={embed_dim}, depths={depths}")
        print(f"✅ 权重加载: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    def forward(self, x, t):
        feats = self.backbone(x, t)
        shared, private = [], []
        for i, f in enumerate(feats):
            s, p = self.split[i](f).chunk(2, dim=1)
            shared.append(s)
            private.append(p)
        return shared, private


class CrossModalDisentangleLoss(nn.Module):
    def __init__(self, lambda_consistency=1.0, lambda_orthogonal=0.1):
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.lambda_orthogonal = lambda_orthogonal

    def forward(self, shared_feats, private_feats):
        loss_consistency = 0
        loss_orthogonal = 0
        modals = list(shared_feats.keys())

        for i in range(len(modals)):
            for j in range(i + 1, len(modals)):
                loss_consistency += F.mse_loss(shared_feats[modals[i]], shared_feats[modals[j]])

        for m in modals:
            s, p = shared_feats[m], private_feats[m]
            s_flat = F.normalize(s.view(s.size(0), -1), dim=1)
            p_flat = F.normalize(p.view(p.size(0), -1), dim=1)
            loss_orthogonal += (s_flat * p_flat).sum(dim=1).pow(2).mean()

        return self.lambda_consistency * loss_consistency + self.lambda_orthogonal * loss_orthogonal


class MultiModalSharedPrivateNet(nn.Module):
    def __init__(self, backbone: str = 'MMSP-tiny', num_classes: int = 25, modals: list = ['image']):
        super().__init__()
        self.modals = modals
        self.encoder = SharedPrivateEncoder(variant=backbone)
        self.fusion_blocks = None   # 延迟构建
        self.decode_head = None
        self.loss_aux = CrossModalDisentangleLoss()

    def init_pretrained(self, path):
        self.encoder.init_pretrained(path)
        dims = self.encoder.out_dims

        # 动态构建 fusion blocks 和 decode head
        self.fusion_blocks = nn.ModuleList([
            MSPABlock(dim=dim) for dim in dims
        ])
        self.decode_head = SegFormerHead(dims, 512, self.decode_head.num_classes if self.decode_head else 25)

    def forward(self, x_list, t=None):
        shared_feats, private_feats = {}, {}
        for i, m in enumerate(self.modals):
            t_modal = torch.ones(x_list[i].shape[0], device=x_list[i].device) * (i / len(self.modals)) if t is None else t
            s, p = self.encoder(x_list[i], t_modal)
            shared_feats[m] = s
            private_feats[m] = p

        fused = []
        for i in range(4):
            x = sum([shared_feats[m][i] for m in self.modals]) / len(self.modals)
            fused.append(self.fusion_blocks[i](x))

        out = self.decode_head(fused)
        out = F.interpolate(out, size=x_list[0].shape[2:], mode='bilinear', align_corners=False)

        if self.training:
            loss_aux = self.loss_aux(
                {k: shared_feats[k][0] for k in self.modals},
                {k: private_feats[k][0] for k in self.modals}
            )
            return out, loss_aux
        return out
