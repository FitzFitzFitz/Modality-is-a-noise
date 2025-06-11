import torch
import torch.nn as nn
import torch.nn.functional as F
from semseg.models.backbones.diffswin import DiffSwinTransformer
from semseg.models.heads import SegFormerHead


def infer_swin_config_from_checkpoint(path: str):
    if isinstance(path, list):
        assert len(path) == 1, f"Expected a single path string, but got a list: {path}"
        path = path[0]

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

    if embed_dim is None:
        raise ValueError("\u274c Failed to parse Swin checkpoint structure")

    return embed_dim, tuple(depths), tuple(num_heads), state_dict


class DiffSwinEncoder(nn.Module):
    def __init__(self, embed_dim=None, depths=None, num_heads=None):
        super().__init__()

        assert embed_dim is not None and depths is not None and num_heads is not None, \
            "embed_dim, depths and num_heads must be provided explicitly."

        self.embed_dim = embed_dim
        self.backbone = DiffSwinTransformer(
            pretrain_img_size=224,
            in_channels=3,
            embed_dims=embed_dim,
            depths=depths,
            num_heads=num_heads,
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            drop_path_rate=0.1,
            patch_norm=True,
            use_abs_pos_embed=False,
            learned_sinusoidal_dim=16,
            norm_cfg=dict(type='LN'),
        )

        self.dims = [embed_dim * 2**i for i in range(4)]
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, dim * 2, kernel_size=1) for dim in self.dims
        ])

    def init_pretrained(self, path):
        if isinstance(path, list):
            assert len(path) == 1, f"Expected single path, got list: {path}"
            path = path[0]
        embed_dim, depths, num_heads, state_dict = infer_swin_config_from_checkpoint(path)
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"\u2705 Swin weights loaded: missing={{len(msg.missing_keys)}}, unexpected={{len(msg.unexpected_keys)}}")

    def forward(self, x, t):
        feats = self.backbone(x, t)
        shared, specific = [], []
        for i, f in enumerate(feats):
            s, p = self.proj[i](f).chunk(2, dim=1)
            shared.append(s)
            specific.append(p)
        return shared, specific


class ModalityDisentangler(nn.Module):
    def __init__(self):
        super().__init__()

    def orthogonal_loss(self, s, p):
        s = F.normalize(s.view(s.size(0), -1), dim=1)
        p = F.normalize(p.view(p.size(0), -1), dim=1)
        return (s * p).sum(dim=1).pow(2).mean()

    def consistency_loss(self, shared_feats):
        loss = 0
        modals = list(shared_feats.keys())
        for i in range(len(modals)):
            for j in range(i+1, len(modals)):
                loss += F.mse_loss(shared_feats[modals[i]], shared_feats[modals[j]])
        return loss

    def forward(self, shared_feats, specific_feats):
        loss_c = self.consistency_loss(shared_feats)
        loss_o = sum([self.orthogonal_loss(shared_feats[k], specific_feats[k]) for k in shared_feats])
        return loss_c + 0.1 * loss_o


class DiffusionFusionDecoder(nn.Module):
    def __init__(self, dims, num_classes):
        super().__init__()
        self.fuse_blocks = nn.ModuleList([
            nn.Conv2d(dim, dim, 1) for dim in dims
        ])
        self.decode_head = SegFormerHead(dims, 512, num_classes)

    def forward(self, shared_feats):
        fused = []
        for i in range(4):
            f = torch.stack([shared_feats[m][i] for m in shared_feats], dim=0).mean(dim=0)
            f = self.fuse_blocks[i](f)
            fused.append(f)
        return self.decode_head(fused)


class DiffModalSegV1(nn.Module):
    def __init__(self, backbone: str = '', num_classes=25, modals=['image', 'depth', 'event']):
        super().__init__()
        self.modals = modals
        self.t_mode = 'permodality'
        self.max_t = 1000

        backbone_settings = {
            'v1-tiny': dict(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)),
            'v1-base': dict(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
        }
        assert backbone in backbone_settings, f"Unsupported backbone: {backbone}"
        settings = backbone_settings[backbone]

        self.encoder = DiffSwinEncoder(
            embed_dim=settings['embed_dim'],
            depths=settings['depths'],
            num_heads=settings['num_heads']
        )
        dims = self.encoder.dims
        self.disentangler = ModalityDisentangler()
        self.decoder = DiffusionFusionDecoder(dims, num_classes)

    def init_pretrained(self, path):
        self.encoder.init_pretrained(path)

    def forward(self, x_list, t=None):
        shared_feats, specific_feats = {}, {}

        if t is None:
            if self.t_mode == 'shared':
                t = torch.randint(0, self.max_t, (1,), device=x_list[0].device).float() / self.max_t
                t = t.expand(x_list[0].shape[0])

        for i, m in enumerate(self.modals):
            if self.t_mode == 'permodality' or t is None:
                t_modal = torch.randint(0, self.max_t, (1,), device=x_list[i].device).float() / self.max_t
                t_modal = t_modal.expand(x_list[i].shape[0])
            else:
                t_modal = t
            s, p = self.encoder(x_list[i], t_modal)
            shared_feats[m] = s
            specific_feats[m] = p

        fused_out = self.decoder({m: shared_feats[m] for m in self.modals})
        fused_out = F.interpolate(fused_out, size=x_list[0].shape[2:], mode='bilinear', align_corners=False)

        if self.training:
            loss = self.disentangler(
                {m: shared_feats[m][0] for m in self.modals},
                {m: specific_feats[m][0] for m in self.modals}
            )
            return fused_out, loss
        return fused_out
    