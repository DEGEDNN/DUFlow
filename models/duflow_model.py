"""
DUFlow: Generative--Discriminative Flow Matching with Topology--Boundary Debate Routing.

This module extends the original dual-branch DUFlow backbone with the complete
paper-level routing path:

1) Discriminative/feature U-Net with four-direction dirGCRU skip enhancement.
2) Conditional Flow-Matching U-Net with an independent encoder.
3) Cross-branch encoder fusion at E3/E4 via SE channel attention.
4) Four decoder fusion points from discriminative decoder features into Flow decoder.
5) One-step Euler reconstruction of the generative mask probability p_g.
6) Training-only analytic topology and boundary teachers.
7) Confidence/relevance-weighted debate target T (detached from gradients).
8) Ground-truth-free convolutional router using only observable quantities.
9) Final routed probability p_f = w_d * p_d + w_g * p_g.
10) Joint objective: flow + discriminative + topology + boundary + route + fused.

IMPORTANT IMPLEMENTATION NOTE
-----------------------------
The paper defines binary crack probabilities p_d and p_g, while the legacy code
uses a 3-channel Flow velocity output. To preserve checkpoint compatibility,
`flow_out_channels` defaults to 3. `flow_to_probability()` converts an Euler-updated
flow state to one crack-probability channel using a configurable reduction:
"mean" (default), "first", or an integer channel index.

The paper specifies the semantics of topology/boundary teachers but does not give
an exact pixelwise implementation of C_top, C_bnd, R_top, and R_bnd. This file uses
stable morphology-based realizations that follow the text:
- topology: soft skeleton disagreement, emphasized near a dilated GT skeleton;
- boundary: contour mismatch + FP/FN terms, emphasized near the GT contour;
- teacher confidence: absolute cost disagreement;
- structural relevance: topology/boundary relevance maps;
- debate target: Eq. T=(R_top V_top + R_bnd V_bnd)/(R_top+R_bnd+eps).
All teacher hyperparameters are exposed for easy replacement if your experimental
implementation used a different exact definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic layers
# -----------------------------------------------------------------------------


def norm_layer(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        norm_layer(out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        norm_layer(out_ch),
        nn.SiLU(),
    )


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# -----------------------------------------------------------------------------
# dirGCRU
# -----------------------------------------------------------------------------


class gCRU(nn.Module):
    """1D gated convolutional recurrent unit."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
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

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)
        gates = self.conv_gate(combined)
        update_gate, reset_gate = torch.sigmoid(gates).chunk(2, dim=1)
        combined_reset = torch.cat([x, reset_gate * h], dim=1)
        h_candidate = torch.tanh(self.conv_out(combined_reset))
        return (1.0 - update_gate) * h + update_gate * h_candidate


class BidirectionalGCRU(nn.Module):
    """Four-directional spatial scan for strengthening U-Net skip tensors."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gru_ver_fwd = gCRU(in_channels, hidden_channels, kernel_size)
        self.gru_ver_bwd = gCRU(in_channels, hidden_channels, kernel_size)
        self.gru_hor_fwd = gCRU(in_channels, hidden_channels, kernel_size)
        self.gru_hor_bwd = gCRU(in_channels, hidden_channels, kernel_size)
        self.output_conv = nn.Conv2d(4 * hidden_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape

        ver_fwd_outs: List[torch.Tensor] = []
        h = torch.zeros(bsz, self.hidden_channels, width, device=x.device, dtype=x.dtype)
        for i in range(height):
            h = self.gru_ver_fwd(x[:, :, i, :], h)
            ver_fwd_outs.append(h)
        ver_fwd = torch.stack(ver_fwd_outs, dim=2)

        ver_bwd_outs: List[torch.Tensor] = []
        h = torch.zeros(bsz, self.hidden_channels, width, device=x.device, dtype=x.dtype)
        for i in range(height - 1, -1, -1):
            h = self.gru_ver_bwd(x[:, :, i, :], h)
            ver_bwd_outs.append(h)
        ver_bwd = torch.stack(ver_bwd_outs[::-1], dim=2)

        x_trans = x.permute(0, 1, 3, 2)

        hor_fwd_outs: List[torch.Tensor] = []
        h = torch.zeros(bsz, self.hidden_channels, height, device=x.device, dtype=x.dtype)
        for i in range(width):
            h = self.gru_hor_fwd(x_trans[:, :, i, :], h)
            hor_fwd_outs.append(h)
        hor_fwd = torch.stack(hor_fwd_outs, dim=2).permute(0, 1, 3, 2)

        hor_bwd_outs: List[torch.Tensor] = []
        h = torch.zeros(bsz, self.hidden_channels, height, device=x.device, dtype=x.dtype)
        for i in range(width - 1, -1, -1):
            h = self.gru_hor_bwd(x_trans[:, :, i, :], h)
            hor_bwd_outs.append(h)
        hor_bwd = torch.stack(hor_bwd_outs[::-1], dim=2).permute(0, 1, 3, 2)

        combined = torch.cat([ver_fwd, ver_bwd, hor_fwd, hor_bwd], dim=1)
        return self.output_conv(combined) + x


# -----------------------------------------------------------------------------
# Discriminative branch
# -----------------------------------------------------------------------------


class FeatureEnhancementUNetParallel(nn.Module):
    """Independent discriminative U-Net producing p_d and multi-scale priors."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_channels: int = 64,
        use_gcru: bool = True,
        active_gcru_levels: Optional[Sequence[int]] = None,
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

        c = base_channels
        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, 2 * c)
        self.enc3 = conv_block(2 * c, 4 * c)
        self.enc4 = conv_block(4 * c, 8 * c)
        self.bottleneck = conv_block(8 * c, 16 * c)

        if use_gcru:
            self.gcru1 = BidirectionalGCRU(c, max(c // 2, 1))
            self.gcru2 = BidirectionalGCRU(2 * c, c)
            self.gcru3 = BidirectionalGCRU(4 * c, 2 * c)
            self.gcru4 = BidirectionalGCRU(8 * c, 4 * c)

        self.up4 = nn.ConvTranspose2d(16 * c, 8 * c, 2, stride=2)
        self.dec4 = conv_block(16 * c, 8 * c)
        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = conv_block(8 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = conv_block(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = conv_block(2 * c, c)

        self.out_conv = nn.Conv2d(c, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def _skip(self, enc_feat: torch.Tensor, gcru_mod: Optional[nn.Module], level: int) -> torch.Tensor:
        if self.use_gcru and level in self.active_gcru_levels:
            if gcru_mod is None:
                raise RuntimeError("dirGCRU module missing while use_gcru=True")
            return gcru_mod(enc_feat)
        return enc_feat

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self._skip(e4, getattr(self, "gcru4", None), 4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self._skip(e3, getattr(self, "gcru3", None), 3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._skip(e2, getattr(self, "gcru2", None), 2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._skip(e1, getattr(self, "gcru1", None), 1)], dim=1))

        p_d = torch.sigmoid(self.out_conv(d1))
        return p_d, [d1, d2, d3, d4], e3, e4


# Backward-compatible alias used by the original code.
FeatureEnhancementUNet_Parallel = FeatureEnhancementUNetParallel


# -----------------------------------------------------------------------------
# Flow branch and cross-branch fusion
# -----------------------------------------------------------------------------


class FeatureFusionModule(nn.Module):
    """Fuse discriminative decoder features into Flow decoder features."""

    def __init__(self, flow_channels: int, feature_channels: int, fusion_type: str = "concat"):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "concat":
            self.fusion_conv = nn.Conv2d(flow_channels + feature_channels, flow_channels, 1)
        elif fusion_type == "add":
            self.align_conv = nn.Conv2d(feature_channels, flow_channels, 1)
        elif fusion_type == "attention":
            self.attention = nn.Sequential(
                nn.Conv2d(flow_channels + feature_channels, flow_channels, 1),
                nn.Sigmoid(),
            )
            self.align_conv = nn.Conv2d(feature_channels, flow_channels, 1)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, flow_feat: torch.Tensor, feature_feat: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            return self.fusion_conv(torch.cat([flow_feat, feature_feat], dim=1))
        if self.fusion_type == "add":
            return flow_feat + self.align_conv(feature_feat)

        combined = torch.cat([flow_feat, feature_feat], dim=1)
        attn = self.attention(combined)
        aligned = self.align_conv(feature_feat)
        return flow_feat * attn + aligned * (1.0 - attn)


class EncoderFusionModule(nn.Module):
    """SE-guided residual injection from discriminative encoder into Flow encoder."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        self.align = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, flow_feat: torch.Tensor, feature_feat: torch.Tensor) -> torch.Tensor:
        gap = feature_feat.mean(dim=(2, 3))
        attn = self.se(gap)[:, :, None, None]
        return flow_feat + self.align(feature_feat) * attn


class ResBlockWithTime(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
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

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class FlowMatchingUNetWithFusion(nn.Module):
    """Independent Flow-Matching U-Net with E3/E4 and four decoder fusion points."""

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        base_channels: int = 64,
        time_embed_dim: Optional[int] = None,
        fusion_type: str = "concat",
    ):
        super().__init__()
        c = base_channels
        time_embed_dim = time_embed_dim or c * 4
        self.base_channels = c

        self.time_embed = nn.Sequential(
            nn.Linear(c, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, 2 * c)
        self.enc3 = conv_block(2 * c, 4 * c)
        self.enc4 = conv_block(4 * c, 8 * c)
        self.bottleneck = conv_block(8 * c, 16 * c)

        self.up4 = nn.ConvTranspose2d(16 * c, 8 * c, 2, stride=2)
        self.dec4 = ResBlockWithTime(16 * c, 8 * c, time_embed_dim)
        self.fuse4 = FeatureFusionModule(8 * c, 8 * c, fusion_type)

        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = ResBlockWithTime(8 * c, 4 * c, time_embed_dim)
        self.fuse3 = FeatureFusionModule(4 * c, 4 * c, fusion_type)

        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = ResBlockWithTime(4 * c, 2 * c, time_embed_dim)
        self.fuse2 = FeatureFusionModule(2 * c, 2 * c, fusion_type)

        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = ResBlockWithTime(2 * c, c, time_embed_dim)
        self.fuse1 = FeatureFusionModule(c, c, fusion_type)

        self.encoder_fuse3 = EncoderFusionModule(4 * c)
        self.encoder_fuse4 = EncoderFusionModule(8 * c)

        self.out_conv = nn.Sequential(
            norm_layer(c),
            nn.SiLU(),
            nn.Conv2d(c, out_channels, 3, padding=1),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        feature_features: Optional[List[Optional[torch.Tensor]]] = None,
        feature_enc3: Optional[torch.Tensor] = None,
        feature_enc4: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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


# -----------------------------------------------------------------------------
# Morphology / edge utilities for teachers and structural losses
# -----------------------------------------------------------------------------


def _ensure_binary_prob(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")
    return x.clamp(0.0, 1.0)


def soft_dilate(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    p = kernel_size // 2
    return F.max_pool2d(x, kernel_size, stride=1, padding=p)


def soft_erode(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size, stride=1, padding=kernel_size // 2)


def soft_open(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    return soft_dilate(soft_erode(x, kernel_size), kernel_size)


def soft_boundary(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Morphological gradient, normalized to [0,1] for a probability mask."""
    return (soft_dilate(x, kernel_size) - soft_erode(x, kernel_size)).clamp(0.0, 1.0)


def soft_skeletonize(x: torch.Tensor, iterations: int = 20, kernel_size: int = 3) -> torch.Tensor:
    """Differentiable-style soft skeleton approximation based on iterative morphology."""
    x = _ensure_binary_prob(x)
    img = x
    skel = F.relu(img - soft_open(img, kernel_size))
    for _ in range(max(iterations - 1, 0)):
        img = soft_erode(img, kernel_size)
        delta = F.relu(img - soft_open(img, kernel_size))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0.0, 1.0)


def image_edge_response(image: torch.Tensor) -> torch.Tensor:
    """Sobel edge magnitude from any C-channel image, returned as [B,1,H,W]."""
    if image.ndim != 4:
        raise ValueError("image must be [B,C,H,W]")
    gray = image.mean(dim=1, keepdim=True)
    dtype, device = gray.dtype, gray.device
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-12)
    denom = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (mag / denom).clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Training-only analytic teachers
# -----------------------------------------------------------------------------


@dataclass
class DebateTeacherConfig:
    temperature: float = 0.5
    skeleton_iterations: int = 12
    skeleton_dilation_kernel: int = 5
    boundary_kernel: int = 3
    boundary_relevance_kernel: int = 5
    confidence_floor: float = 1e-3
    relevance_floor: float = 1e-3
    eps: float = 1e-6


class TopologyTeacher(nn.Module):
    """Analytic teacher emphasizing missed/spurious skeleton responses."""

    def __init__(self, config: Optional[DebateTeacherConfig] = None):
        super().__init__()
        self.cfg = config or DebateTeacherConfig()

    def cost_and_relevance(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = _ensure_binary_prob(pred)
        target = _ensure_binary_prob(target)
        pred_skel = soft_skeletonize(pred, self.cfg.skeleton_iterations)
        with torch.no_grad():
            target_skel = soft_skeletonize(target, self.cfg.skeleton_iterations)
            relevance = soft_dilate(target_skel, self.cfg.skeleton_dilation_kernel)
            relevance = relevance.clamp(0.0, 1.0)

        # Symmetric skeleton disagreement captures missed and spurious centerlines.
        missed = target_skel * (1.0 - pred_skel)
        spurious = pred_skel * (1.0 - target_skel)
        cost = missed + spurious
        return cost, relevance


class BoundaryTeacher(nn.Module):
    """Analytic teacher emphasizing contour mismatch, FP spill, and FN erosion."""

    def __init__(self, config: Optional[DebateTeacherConfig] = None):
        super().__init__()
        self.cfg = config or DebateTeacherConfig()

    def cost_and_relevance(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = _ensure_binary_prob(pred)
        target = _ensure_binary_prob(target)
        pred_b = soft_boundary(pred, self.cfg.boundary_kernel)
        with torch.no_grad():
            target_b = soft_boundary(target, self.cfg.boundary_kernel)
            relevance = soft_dilate(target_b, self.cfg.boundary_relevance_kernel)
            relevance = relevance.clamp(0.0, 1.0)

        contour = (pred_b - target_b).abs()
        false_pos = pred * (1.0 - target)
        false_neg = target * (1.0 - pred)
        # Keep local cost bounded and interpretable.
        cost = (contour + 0.5 * false_pos + 0.5 * false_neg).clamp(0.0, 2.0)
        return cost, relevance


class TopologyBoundaryDebateTeacher(nn.Module):
    """Construct detached soft routing targets from topology/boundary teacher votes."""

    def __init__(self, config: Optional[DebateTeacherConfig] = None):
        super().__init__()
        self.cfg = config or DebateTeacherConfig()
        self.topology = TopologyTeacher(self.cfg)
        self.boundary = BoundaryTeacher(self.cfg)

    def _votes(self, c_d: torch.Tensor, c_g: torch.Tensor) -> torch.Tensor:
        costs = torch.cat([c_d, c_g], dim=1)  # [B,2,H,W]
        return torch.softmax(-costs / self.cfg.temperature, dim=1)

    def _reliability(
        self,
        c_d: torch.Tensor,
        c_g: torch.Tensor,
        relevance: torch.Tensor,
    ) -> torch.Tensor:
        confidence = (c_d - c_g).abs()
        return (
            (confidence + self.cfg.confidence_floor)
            * (relevance + self.cfg.relevance_floor)
        )

    @torch.no_grad()
    def forward(self, p_d: torch.Tensor, p_g: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        top_d, top_rel = self.topology.cost_and_relevance(p_d, target)
        top_g, _ = self.topology.cost_and_relevance(p_g, target)
        bnd_d, bnd_rel = self.boundary.cost_and_relevance(p_d, target)
        bnd_g, _ = self.boundary.cost_and_relevance(p_g, target)

        v_top = self._votes(top_d, top_g)
        v_bnd = self._votes(bnd_d, bnd_g)
        r_top = self._reliability(top_d, top_g, top_rel)
        r_bnd = self._reliability(bnd_d, bnd_g, bnd_rel)

        target_router = (
            r_top * v_top + r_bnd * v_bnd
        ) / (r_top + r_bnd + self.cfg.eps)
        target_router = target_router / target_router.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps)

        return {
            "target": target_router.detach(),
            "v_top": v_top,
            "v_bnd": v_bnd,
            "r_top": r_top,
            "r_bnd": r_bnd,
            "top_cost_d": top_d,
            "top_cost_g": top_g,
            "bnd_cost_d": bnd_d,
            "bnd_cost_g": bnd_g,
        }


# -----------------------------------------------------------------------------
# Deployable ground-truth-free router
# -----------------------------------------------------------------------------


class DebateRouter(nn.Module):
    """Ground-truth-free spatial router W=[w_d,w_g]."""

    def __init__(self, hidden_channels: int = 32, router_temperature: float = 1.0):
        super().__init__()
        self.router_temperature = router_temperature
        # Inputs: image edge, p_d, p_g, |p_d-p_g|, U_d, U_g => 6 channels.
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden_channels, 3, padding=1),
            norm_layer(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            norm_layer(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 1),
        )

    def build_inputs(
        self,
        image: torch.Tensor,
        p_d: torch.Tensor,
        p_g: torch.Tensor,
    ) -> torch.Tensor:
        p_d = _ensure_binary_prob(p_d)
        p_g = _ensure_binary_prob(p_g)
        edge = image_edge_response(image)
        disagreement = (p_d - p_g).abs()
        u_d = 4.0 * p_d * (1.0 - p_d)
        u_g = 4.0 * p_g * (1.0 - p_g)
        return torch.cat([edge, p_d, p_g, disagreement, u_d, u_g], dim=1)

    def forward(
        self,
        image: torch.Tensor,
        p_d: torch.Tensor,
        p_g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        router_input = self.build_inputs(image, p_d, p_g)
        logits = self.net(router_input)
        weights = torch.softmax(logits / self.router_temperature, dim=1)
        return weights, logits


# -----------------------------------------------------------------------------
# Complete DUFlow model
# -----------------------------------------------------------------------------


FlowReduction = Union[str, int]


class DualBranchParallelFlowModel(nn.Module):
    """
    Complete DUFlow model with debate-routed expert collaboration.

    Backward-compatible backbone call:
        v_pred, p_d = model(x, timesteps)

    Full routed call:
        out = model.forward_routed(
            x, timesteps, flow_state=x_t, router_image=condition_image,
            target_mask=G  # only during training if teacher target is desired
        )

    Returned `out` contains v_pred, p_d, p_g, router_weights, and p_f.
    """

    def __init__(
        self,
        feature_unet: Optional[nn.Module] = None,
        flow_unet: Optional[nn.Module] = None,
        in_channels: int = 10,
        feature_in_channels: int = 6,
        feature_slice: Tuple[int, int] = (3, 9),
        flow_out_channels: int = 3,
        base_channels: int = 64,
        fusion_type: str = "concat",
        use_gcru: bool = True,
        active_gcru_levels: Optional[Sequence[int]] = None,
        flow_probability_reduction: FlowReduction = "mean",
        router_hidden_channels: int = 32,
        teacher_config: Optional[DebateTeacherConfig] = None,
    ):
        super().__init__()
        self.feature_slice = feature_slice
        self.feature_in_channels = feature_in_channels
        self.flow_out_channels = flow_out_channels
        self.flow_probability_reduction = flow_probability_reduction

        self.feature_unet = feature_unet or FeatureEnhancementUNetParallel(
            in_channels=feature_in_channels,
            out_channels=1,
            base_channels=base_channels,
            use_gcru=use_gcru,
            active_gcru_levels=active_gcru_levels,
        )
        self.flow_unet = flow_unet or FlowMatchingUNetWithFusion(
            in_channels=in_channels,
            out_channels=flow_out_channels,
            base_channels=base_channels,
            fusion_type=fusion_type,
        )
        self.debate_teacher = TopologyBoundaryDebateTeacher(teacher_config)
        self.router = DebateRouter(hidden_channels=router_hidden_channels)

    def extract_condition(self, x: torch.Tensor) -> torch.Tensor:
        s, e = self.feature_slice
        condition = x[:, s:e, :, :]
        if condition.shape[1] != self.feature_in_channels:
            raise ValueError(
                f"feature_slice={self.feature_slice} yields {condition.shape[1]} channels; "
                f"expected {self.feature_in_channels}."
            )
        return condition

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy backbone interface: returns velocity and discriminative probability."""
        condition = self.extract_condition(x)
        p_d, multi_scale_features, feature_enc3, feature_enc4 = self.feature_unet(condition)
        v_pred = self.flow_unet(
            x,
            timesteps,
            multi_scale_features,
            feature_enc3,
            feature_enc4,
        )
        return v_pred, p_d

    @staticmethod
    def euler_step(
        flow_state: torch.Tensor,
        velocity: torch.Tensor,
        dt: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        if flow_state.shape != velocity.shape:
            raise ValueError(
                "Euler state and velocity must have the same shape. "
                f"Got state={tuple(flow_state.shape)}, velocity={tuple(velocity.shape)}"
            )
        if not torch.is_tensor(dt):
            dt = torch.tensor(float(dt), device=flow_state.device, dtype=flow_state.dtype)
        while dt.ndim < flow_state.ndim:
            dt = dt[..., None]
        return flow_state + dt * velocity

    def flow_to_probability(self, flow_mask_state: torch.Tensor) -> torch.Tensor:
        """Map Euler-updated Flow state to binary crack probability p_g."""
        if flow_mask_state.ndim != 4:
            raise ValueError("flow_mask_state must be [B,C,H,W]")

        reduction = self.flow_probability_reduction
        if isinstance(reduction, int):
            logits = flow_mask_state[:, reduction : reduction + 1]
        elif reduction == "first":
            logits = flow_mask_state[:, :1]
        elif reduction == "mean":
            logits = flow_mask_state.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported flow_probability_reduction={reduction!r}")
        return torch.sigmoid(logits)

    def forward_routed(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        flow_state: torch.Tensor,
        router_image: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        dt: Union[float, torch.Tensor] = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete one-step DUFlow path.

        Args:
            x: preassembled Flow-network input [B,in_channels,H,W].
            timesteps: current ODE/FM time [B].
            flow_state: x_t (or z for t=0) with the same channels as v_pred.
            router_image: image used to compute observable Sobel edge response.
                Defaults to the discriminative condition extracted from x.
            target_mask: optional GT [B,1,H,W]. If supplied, training-only teacher
                outputs and detached routing target are returned.
            dt: Euler step size; one-step inference normally uses 1.0 from t=0 to 1.
        """
        v_pred, p_d = self.forward(x, timesteps)
        x_next = self.euler_step(flow_state, v_pred, dt=dt)
        p_g = self.flow_to_probability(x_next)

        if router_image is None:
            router_image = self.extract_condition(x)
        weights, router_logits = self.router(router_image, p_d, p_g)
        w_d, w_g = weights[:, 0:1], weights[:, 1:2]
        p_f = w_d * p_d + w_g * p_g

        out: Dict[str, torch.Tensor] = {
            "v_pred": v_pred,
            "p_d": p_d,
            "p_g": p_g,
            "p_f": p_f,
            "router_weights": weights,
            "router_logits": router_logits,
            "w_d": w_d,
            "w_g": w_g,
            "flow_state_next": x_next,
        }
        if target_mask is not None:
            teacher = self.debate_teacher(p_d.detach(), p_g.detach(), target_mask.detach())
            out["router_target"] = teacher.pop("target")
            out.update({f"teacher_{k}": v for k, v in teacher.items()})
        return out

    @torch.no_grad()
    def one_step_inference(
        self,
        x: torch.Tensor,
        noise_state: torch.Tensor,
        router_image: Optional[torch.Tensor] = None,
        t0: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Paper-style deterministic one-step Euler inference from noise z at t=0."""
        timesteps = torch.full(
            (x.shape[0],),
            float(t0),
            device=x.device,
            dtype=x.dtype,
        )
        return self.forward_routed(
            x=x,
            timesteps=timesteps,
            flow_state=noise_state,
            router_image=router_image,
            target_mask=None,
            dt=1.0 - t0,
        )

    def prepare_cuda_inference(self, example_input, example_timesteps, warmup: int = 3):
        """Backward-compatible optional CUDA graph helper if present in the package."""
        from .dirgcru_inference import prepare_cuda_inference

        return prepare_cuda_inference(self, example_input, example_timesteps, warmup)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------


class BceDiceLoss(nn.Module):
    """Binary BCE + Dice loss. Defaults match the paper's unweighted sum."""

    def __init__(self, weight_bce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCELoss()

    @staticmethod
    def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        loss = 1.0 - (2.0 * intersection + smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + smooth
        )
        return loss.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice_loss(pred, target)


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class SoftTopologyLoss(nn.Module):
    """Skeleton-overlap structural loss, suitable for fused and branch supervision."""

    def __init__(self, iterations: int = 12, smooth: float = 1e-5):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sp = soft_skeletonize(pred, self.iterations)
        st = soft_skeletonize(target, self.iterations)
        dims = (1, 2, 3)
        tprec = (sp * target).sum(dims) + self.smooth
        tprec = tprec / (sp.sum(dims) + self.smooth)
        tsens = (st * pred).sum(dims) + self.smooth
        tsens = tsens / (st.sum(dims) + self.smooth)
        cldice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return (1.0 - cldice).mean()


class SoftBoundaryLoss(nn.Module):
    """Boundary Dice loss using a soft morphological gradient."""

    def __init__(self, kernel_size: int = 3, smooth: float = 1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bp = soft_boundary(pred, self.kernel_size).flatten(1)
        bt = soft_boundary(target, self.kernel_size).flatten(1)
        inter = (bp * bt).sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (bp.sum(dim=1) + bt.sum(dim=1) + self.smooth)
        return (1.0 - dice).mean()


class SoftRoutingCrossEntropy(nn.Module):
    """L_route = -mean(sum_k T_k log(W_k+eps))."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -(target * torch.log(weights.clamp_min(self.eps))).sum(dim=1).mean()


class DUFlowJointLoss(nn.Module):
    """
    Complete paper-style joint DUFlow objective.

    L_total = L_flow + lambda_d L_d + lambda_t L_top + lambda_b L_bnd
              + lambda_r L_route + lambda_f L_fused

    Topology/boundary objectives are applied strongly to p_f and with lower weight
    to individual p_d/p_g, matching the paper's description.
    """

    def __init__(
        self,
        lambda_d: float = 1.0,
        lambda_t: float = 0.05,
        lambda_b: float = 0.05,
        lambda_r: float = 0.10,
        lambda_f: float = 1.0,
        branch_structural_weight: float = 0.25,
        skeleton_iterations: int = 12,
    ):
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.lambda_b = lambda_b
        self.lambda_r = lambda_r
        self.lambda_f = lambda_f
        self.branch_structural_weight = branch_structural_weight

        self.seg_loss = BceDiceLoss(1.0, 1.0)
        self.topology_loss = SoftTopologyLoss(iterations=skeleton_iterations)
        self.boundary_loss = SoftBoundaryLoss(kernel_size=3)
        self.route_loss = SoftRoutingCrossEntropy()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        u_t: torch.Tensor,
        target_mask: torch.Tensor,
        router_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        required = ("v_pred", "p_d", "p_g", "p_f", "router_weights")
        missing = [k for k in required if k not in outputs]
        if missing:
            raise KeyError(f"Missing DUFlow outputs: {missing}")

        v_pred = outputs["v_pred"]
        p_d = outputs["p_d"]
        p_g = outputs["p_g"]
        p_f = outputs["p_f"]
        weights = outputs["router_weights"]

        loss_flow = mean_flat((v_pred - u_t) ** 2).mean()
        loss_d = self.seg_loss(p_d, target_mask)
        loss_fused = self.seg_loss(p_f, target_mask)

        top_f = self.topology_loss(p_f, target_mask)
        top_d = self.topology_loss(p_d, target_mask)
        top_g = self.topology_loss(p_g, target_mask)
        loss_top = top_f + self.branch_structural_weight * 0.5 * (top_d + top_g)

        bnd_f = self.boundary_loss(p_f, target_mask)
        bnd_d = self.boundary_loss(p_d, target_mask)
        bnd_g = self.boundary_loss(p_g, target_mask)
        loss_bnd = bnd_f + self.branch_structural_weight * 0.5 * (bnd_d + bnd_g)

        if router_target is None:
            router_target = outputs.get("router_target")
        if router_target is None:
            raise ValueError(
                "router_target is required for L_route. Pass target_mask to model.forward_routed() "
                "or supply router_target explicitly."
            )
        loss_route = self.route_loss(weights, router_target.detach())

        loss_total = (
            loss_flow
            + self.lambda_d * loss_d
            + self.lambda_t * loss_top
            + self.lambda_b * loss_bnd
            + self.lambda_r * loss_route
            + self.lambda_f * loss_fused
        )

        return {
            "loss": loss_total,
            "loss_flow": loss_flow,
            "loss_d": loss_d,
            "loss_top": loss_top,
            "loss_bnd": loss_bnd,
            "loss_route": loss_route,
            "loss_fused": loss_fused,
            "top_fused": top_f,
            "top_discriminative": top_d,
            "top_generative": top_g,
            "bnd_fused": bnd_f,
            "bnd_discriminative": bnd_d,
            "bnd_generative": bnd_g,
        }


# Backward-compatible loss alias for old training scripts.
class DualBranchLoss(nn.Module):
    """Legacy flow + discriminative BCE/Dice loss from the earlier model file."""

    def __init__(self, weight_cal: float = 10.0):
        super().__init__()
        self.weight_cal = weight_cal
        self.bce_dice_loss = BceDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def forward(
        self,
        v_pred: torch.Tensor,
        u_t: torch.Tensor,
        cal: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_diff = mean_flat((v_pred - u_t) ** 2)
        loss_cal = self.bce_dice_loss(cal, target_mask)
        loss_total = loss_diff + self.weight_cal * loss_cal
        return {
            "loss": loss_total.mean(),
            "loss_diff": loss_diff.mean(),
            "loss_cal": loss_cal,
        }


# -----------------------------------------------------------------------------
# Flow-Matching training helpers
# -----------------------------------------------------------------------------


def sample_flow_matching_state(
    target_mask: torch.Tensor,
    flow_channels: int = 3,
    timesteps: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct x_t=(1-t)z+tG and u_t=G-z.

    If flow_channels>1, the single-channel GT mask is repeated across channels,
    matching legacy 3-channel Flow outputs.
    """
    target_mask = _ensure_binary_prob(target_mask)
    b = target_mask.shape[0]
    target_flow = target_mask.repeat(1, flow_channels, 1, 1)
    if timesteps is None:
        timesteps = torch.rand(b, device=target_mask.device, dtype=target_mask.dtype)
    if noise is None:
        noise = torch.randn_like(target_flow)
    t = timesteps[:, None, None, None]
    x_t = (1.0 - t) * noise + t * target_flow
    u_t = target_flow - noise
    return x_t, u_t, noise, timesteps


# -----------------------------------------------------------------------------
# Minimal smoke test
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Use a tiny configuration for a quick CPU smoke test. Production paper config
    # remains base_channels=64 and typically 256x256 or the dataset crop size.
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualBranchParallelFlowModel(
        in_channels=10,
        feature_in_channels=6,
        feature_slice=(3, 9),
        flow_out_channels=3,
        base_channels=8,
        fusion_type="concat",
        use_gcru=False,  # enable in real DUFlow; disabled only to keep smoke test fast
        flow_probability_reduction="mean",
    ).to(device)

    b, h, w = 2, 32, 32
    target = torch.randint(0, 2, (b, 1, h, w), device=device).float()
    x_t, u_t, _, t = sample_flow_matching_state(target, flow_channels=3)

    # Legacy code expects a preassembled 10-channel tensor. Here we construct a
    # synthetic one where channels 0:3 hold x_t and 3:9 hold six condition channels.
    condition = torch.randn(b, 6, h, w, device=device)
    extra = torch.randn(b, 1, h, w, device=device)
    flow_input = torch.cat([x_t, condition, extra], dim=1)

    out = model.forward_routed(
        x=flow_input,
        timesteps=t,
        flow_state=x_t,
        router_image=condition,
        target_mask=target,
        dt=1.0 - t,
    )

    loss_fn = DUFlowJointLoss(
        lambda_d=1.0,
        lambda_t=0.05,
        lambda_b=0.05,
        lambda_r=0.10,
        lambda_f=1.0,
    ).to(device)
    losses = loss_fn(out, u_t=u_t, target_mask=target)

    print("DUFlow smoke test")
    print("v_pred:", tuple(out["v_pred"].shape))
    print("p_d:", tuple(out["p_d"].shape))
    print("p_g:", tuple(out["p_g"].shape))
    print("router_weights:", tuple(out["router_weights"].shape))
    print("p_f:", tuple(out["p_f"].shape))
    print("loss:", float(losses["loss"].detach().cpu()))
