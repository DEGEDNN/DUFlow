"""
Dual-branch flow matching model with independent encoders and multi-scale fusion.

Feature branch:
    condition image -> segmentation prior cal + decoder features [d1, d2, d3, d4]

Flow branch:
    full flow input -> independent flow encoder/decoder -> velocity field

The two branches do not share an encoder. The flow decoder fuses feature-branch
decoder features at matching spatial scales.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class gCRU(nn.Module):
    """1D gated convolutional recurrent unit."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv_gate = nn.Conv1d(
            in_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size,
            padding=padding,
        )
        self.conv_out = nn.Conv1d(
            in_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv_gate(combined)
        update_gate, reset_gate = torch.sigmoid(gates).chunk(2, dim=1)
        combined_reset = torch.cat([x, reset_gate * h], dim=1)
        h_candidate = torch.tanh(self.conv_out(combined_reset))
        return (1 - update_gate) * h + update_gate * h_candidate


class BidirectionalGCRU(nn.Module):
    """Four-directional spatial scan used to strengthen feature skip tensors."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gru_ver_fwd = gCRU(in_channels, hidden_channels, kernel_size)
        self.gru_ver_bwd = gCRU(in_channels, hidden_channels, kernel_size)
        self.gru_hor_fwd = gCRU(in_channels, hidden_channels, kernel_size)
        self.gru_hor_bwd = gCRU(in_channels, hidden_channels, kernel_size)
        self.output_conv = nn.Conv2d(4 * hidden_channels, in_channels, kernel_size=1)

    def forward(self, x):
        bsz, _, height, width = x.shape

        ver_fwd_outs = []
        h = torch.zeros(bsz, self.hidden_channels, width, device=x.device, dtype=x.dtype)
        for i in range(height):
            h = self.gru_ver_fwd(x[:, :, i, :], h)
            ver_fwd_outs.append(h)
        ver_fwd_outs = torch.stack(ver_fwd_outs, dim=2)

        ver_bwd_outs = []
        h = torch.zeros(bsz, self.hidden_channels, width, device=x.device, dtype=x.dtype)
        for i in range(height - 1, -1, -1):
            h = self.gru_ver_bwd(x[:, :, i, :], h)
            ver_bwd_outs.insert(0, h)
        ver_bwd_outs = torch.stack(ver_bwd_outs, dim=2)

        x_trans = x.permute(0, 1, 3, 2)
        hor_fwd_outs = []
        h = torch.zeros(bsz, self.hidden_channels, height, device=x.device, dtype=x.dtype)
        for i in range(width):
            h = self.gru_hor_fwd(x_trans[:, :, i, :], h)
            hor_fwd_outs.append(h)
        hor_fwd_outs = torch.stack(hor_fwd_outs, dim=2).permute(0, 1, 3, 2)

        hor_bwd_outs = []
        h = torch.zeros(bsz, self.hidden_channels, height, device=x.device, dtype=x.dtype)
        for i in range(width - 1, -1, -1):
            h = self.gru_hor_bwd(x_trans[:, :, i, :], h)
            hor_bwd_outs.insert(0, h)
        hor_bwd_outs = torch.stack(hor_bwd_outs, dim=2).permute(0, 1, 3, 2)

        combined = torch.cat(
            [ver_fwd_outs, ver_bwd_outs, hor_fwd_outs, hor_bwd_outs],
            dim=1,
        )
        return self.output_conv(combined) + x


def norm_layer(channels):
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        norm_layer(out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        norm_layer(out_ch),
        nn.SiLU(),
    )


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FeatureEnhancementUNet_Parallel(nn.Module):
    """
    Feature branch with an independent UNet encoder.

    Returns:
        cal: [B, 1, H, W]
        decoder_features: [d1, d2, d3, d4]
            d1: [B, C, H, W]
            d2: [B, 2C, H/2, W/2]
            d3: [B, 4C, H/4, W/4]
            d4: [B, 8C, H/8, W/8]
        e3: [B, 4C, H/4, W/4] – encoder feature for cross-branch fusion
        e4: [B, 8C, H/8, W/8] – encoder feature for cross-branch fusion
    """

    def __init__(
        self,
        in_channels=6,
        out_channels=1,
        base_channels=64,
        use_gcru=True,
        active_gcru_levels=None,
    ):
        super().__init__()
        self.use_gcru = use_gcru
        self.active_gcru_levels = (
            frozenset((1, 2, 3, 4))
            if active_gcru_levels is None
            else frozenset(active_gcru_levels)
        )
        invalid_levels = self.active_gcru_levels.difference((1, 2, 3, 4))
        if invalid_levels:
            raise ValueError(f"Invalid dirGCRU levels: {sorted(invalid_levels)}")
        self.enc1 = conv_block(in_channels, base_channels)
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = conv_block(base_channels * 4, base_channels * 8)
        self.bottleneck = conv_block(base_channels * 8, base_channels * 16)

        if use_gcru:
            self.gcru1 = BidirectionalGCRU(base_channels, base_channels // 2)
            self.gcru2 = BidirectionalGCRU(base_channels * 2, base_channels)
            self.gcru3 = BidirectionalGCRU(base_channels * 4, base_channels * 2)
            self.gcru4 = BidirectionalGCRU(base_channels * 8, base_channels * 4)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = conv_block(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = conv_block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = conv_block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = conv_block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def _skip(self, enc_feat, gcru_mod, level):
        """use_gcru=True 时经过4向空间扫描，否则直接透传 encoder 特征。"""
        return (
            gcru_mod(enc_feat)
            if self.use_gcru and level in self.active_gcru_levels
            else enc_feat
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        gcru1 = getattr(self, "gcru1", None)
        gcru2 = getattr(self, "gcru2", None)
        gcru3 = getattr(self, "gcru3", None)
        gcru4 = getattr(self, "gcru4", None)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self._skip(e4, gcru4, 4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self._skip(e3, gcru3, 3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._skip(e2, gcru2, 2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._skip(e1, gcru1, 1)], dim=1))

        cal = torch.sigmoid(self.out_conv(d1))
        return cal, [d1, d2, d3, d4], e3, e4


class FeatureFusionModule(nn.Module):
    """Fuse feature-branch tensors into flow-branch decoder tensors."""

    def __init__(self, flow_channels, feature_channels, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.fusion_conv = nn.Conv2d(flow_channels + feature_channels, flow_channels, 1)
        elif fusion_type == 'add':
            self.align_conv = nn.Conv2d(feature_channels, flow_channels, 1)
        elif fusion_type == 'attention':
            self.attention = nn.Sequential(
                nn.Conv2d(flow_channels + feature_channels, flow_channels, 1),
                nn.Sigmoid(),
            )
            self.align_conv = nn.Conv2d(feature_channels, flow_channels, 1)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, flow_feat, feature_feat):
        if self.fusion_type == 'concat':
            return self.fusion_conv(torch.cat([flow_feat, feature_feat], dim=1))
        if self.fusion_type == 'add':
            return flow_feat + self.align_conv(feature_feat)

        combined = torch.cat([flow_feat, feature_feat], dim=1)
        attn = self.attention(combined)
        aligned = self.align_conv(feature_feat)
        return flow_feat * attn + aligned * (1 - attn)


class EncoderFusionModule(nn.Module):
    """用特征分支的 encoder feature 通过 SE 通道注意力注入流分支，残差叠加。

    feature_feat 经 GAP 压缩为全局描述子，再经两层 FC 生成 [0,1] 的通道权重，
    对 feature_feat 做重标定后以残差方式加到 flow_feat 上。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        # 两层 FC 做 squeeze-excitation，对 feature 特征生成通道权重
        self.se = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        # 对齐卷积：保持通道数，明确语义隔离（不与 decoder 融合共享权重）
        self.align = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, flow_feat, feature_feat):
        # [B, C] 全局平均池化
        gap = feature_feat.mean(dim=[2, 3])
        # [B, C] 通道注意力权重
        attn = self.se(gap)[:, :, None, None]
        # 重标定后的 feature 特征，残差加到 flow 特征
        return flow_feat + self.align(feature_feat) * attn


class ResBlockWithTime(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.norm1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.norm2 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class FlowMatchingUNetWithFusion(nn.Module):
    """Flow branch with independent encoder, four decoder fusion points, and two encoder fusion points (enc3 + enc4)."""

    def __init__(
        self,
        in_channels=10,
        out_channels=3,
        base_channels=64,
        time_embed_dim=None,
        fusion_type='concat',
    ):
        super().__init__()
        time_embed_dim = time_embed_dim or base_channels * 4
        self.base_channels = base_channels

        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.enc1 = conv_block(in_channels, base_channels)
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = conv_block(base_channels * 4, base_channels * 8)
        self.bottleneck = conv_block(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ResBlockWithTime(base_channels * 16, base_channels * 8, time_embed_dim)
        self.fuse4 = FeatureFusionModule(base_channels * 8, base_channels * 8, fusion_type)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ResBlockWithTime(base_channels * 8, base_channels * 4, time_embed_dim)
        self.fuse3 = FeatureFusionModule(base_channels * 4, base_channels * 4, fusion_type)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ResBlockWithTime(base_channels * 4, base_channels * 2, time_embed_dim)
        self.fuse2 = FeatureFusionModule(base_channels * 2, base_channels * 2, fusion_type)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ResBlockWithTime(base_channels * 2, base_channels, time_embed_dim)
        self.fuse1 = FeatureFusionModule(base_channels, base_channels, fusion_type)

        # encoder 跨分支融合（判别 enc → flow enc，channel attention）
        self.encoder_fuse3 = EncoderFusionModule(base_channels * 4)
        self.encoder_fuse4 = EncoderFusionModule(base_channels * 8)

        self.out_conv = nn.Sequential(
            norm_layer(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, timesteps, feature_features: Optional[List[torch.Tensor]] = None,
                feature_enc3: Optional[torch.Tensor] = None, feature_enc4: Optional[torch.Tensor] = None):
        t_emb = timestep_embedding(timesteps, self.base_channels)
        t_emb = self.time_embed(t_emb)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        if feature_enc3 is not None:
            e3 = self.encoder_fuse3(e3, feature_enc3)
        e4 = self.enc4(self.pool(e3))
        if feature_enc4 is not None:
            e4 = self.encoder_fuse4(e4, feature_enc4)
        b = self.bottleneck(self.pool(e4))

        if feature_features is None:
            feature_features = [None, None, None, None]
        fd1, fd2, fd3, fd4 = feature_features

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1), t_emb)
        if fd4 is not None:
            d4 = self.fuse4(d4, fd4)

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)
        if fd3 is not None:
            d3 = self.fuse3(d3, fd3)

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)
        if fd2 is not None:
            d2 = self.fuse2(d2, fd2)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)
        if fd1 is not None:
            d1 = self.fuse1(d1, fd1)

        return self.out_conv(d1)


class DualBranchParallelFlowModel(nn.Module):
    """Independent-encoder dual-branch model with real multi-scale fusion."""

    def __init__(
        self,
        feature_unet=None,
        flow_unet=None,
        in_channels=10,
        base_channels=64,
        fusion_type='concat',
        use_gcru=True,
        active_gcru_levels=None,
    ):
        super().__init__()
        self.feature_unet = feature_unet or FeatureEnhancementUNet_Parallel(
            in_channels=6,
            out_channels=1,
            base_channels=base_channels,
            use_gcru=use_gcru,
            active_gcru_levels=active_gcru_levels,
        )
        self.flow_unet = flow_unet or FlowMatchingUNetWithFusion(
            in_channels=in_channels,
            out_channels=3,
            base_channels=base_channels,
            fusion_type=fusion_type,
        )

    def forward(self, x, timesteps):
        condition = x[:, 3:9, :, :]
        cal, multi_scale_features, feature_enc3, feature_enc4 = self.feature_unet(condition)
        v_pred = self.flow_unet(x, timesteps, multi_scale_features, feature_enc3, feature_enc4)
        return v_pred, cal

    def prepare_cuda_inference(self, example_input, example_timesteps, warmup=3):
        """Create a fixed-shape GPU runner after loading weights and calling eval().

        Use the returned runner(x, t) for graph replay; ordinary forward and
        training remain available. See dirgcru_inference for lifecycle constraints.
        """
        from .dirgcru_inference import prepare_cuda_inference

        return prepare_cuda_inference(self, example_input, example_timesteps, warmup)


class BceDiceLoss(nn.Module):
    """BCE + Dice loss for the feature branch mask."""

    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCELoss()

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def forward(self, pred, target):
        return (
            self.weight_bce * self.bce(pred, target)
            + self.weight_dice * self.dice_loss(pred, target)
        )


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class DualBranchLoss(nn.Module):
    """Flow matching loss plus supervised feature-branch mask loss."""

    def __init__(self, weight_cal=10.0):
        super().__init__()
        self.weight_cal = weight_cal
        self.bce_dice_loss = BceDiceLoss()

    def forward(self, v_pred, u_t, cal, target_mask):
        loss_diff = mean_flat((v_pred - u_t) ** 2)
        loss_cal = self.bce_dice_loss(cal, target_mask)
        loss_total = loss_diff + self.weight_cal * loss_cal
        return {
            'loss': loss_total.mean(),
            'loss_diff': loss_diff.mean(),
            'loss_cal': loss_cal,
        }


if __name__ == "__main__":
    model = DualBranchParallelFlowModel(base_channels=64, fusion_type='concat')
    x = torch.randn(2, 10, 256, 256)
    timesteps = torch.rand(2)
    v_pred, cal = model(x, timesteps)
    print(f"v_pred shape: {v_pred.shape}")
    print(f"cal shape: {cal.shape}")

    loss_fn = DualBranchLoss(weight_cal=10.0)
    losses = loss_fn(
        v_pred,
        torch.randn_like(v_pred),
        cal,
        torch.randint(0, 2, cal.shape).float(),
    )
    print(f"total loss: {losses['loss']:.4f}")
    print(f"flow loss: {losses['loss_diff']:.4f}")
    print(f"cal loss: {losses['loss_cal']:.4f}")
