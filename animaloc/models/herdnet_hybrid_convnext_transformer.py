"""
HerdNet Hybrid: ConvNeXt Stem + Transformer Blocks

Combines CNN spatial precision with Transformer global reasoning:
- ConvNeXt stages 0-1: Preserve fine spatial detail for small objects (40-50px)
- Transformer blocks at 1/8 resolution: Global context and long-range dependencies
- ConvNeXt stages 2-3 replaced by transformer blocks
- FPN decoder + multi-scale detection head (reused from CamouflageHerdNetConvNeXt)

Why this works for small animal detection:
1. CNN stem preserves 40-50px objects as 5-6px features at 1/8 res
2. Transformer sees all spatial locations simultaneously (is this near water? near other iguanas?)
3. No patch tokenization loss — spatial detail enters the transformer already processed
4. ConvNeXt pretrained weights initialize the stem, transformer blocks train from scratch
   or can be initialized from DINOv2 weights
"""

from typing import Optional, List, Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .register import MODELS


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm, matching DINOv2/ViT style."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm transformer block
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialTransformerEncoder(nn.Module):
    """Applies transformer blocks to 2D spatial feature maps.

    Flattens spatial dims to sequence, adds learnable positional embeddings,
    runs through transformer blocks, then reshapes back to spatial.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        max_spatial: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.pos_embed = nn.Parameter(torch.zeros(1, max_spatial * max_spatial, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spatial feature map
        Returns:
            [B, C, H, W] transformed feature map
        """
        B, C, H, W = x.shape
        assert C == self.dim, f"Channel mismatch: got {C}, expected {self.dim}"

        # Flatten spatial to sequence: [B, C, H, W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)

        # Add positional embeddings (interpolate if needed)
        seq_len = H * W
        if seq_len != self.pos_embed.shape[1]:
            pos = self.pos_embed.reshape(1, int(self.pos_embed.shape[1] ** 0.5),
                                         int(self.pos_embed.shape[1] ** 0.5), self.dim)
            pos = pos.permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(H, W), mode='bilinear', align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)
            x = x + pos
        else:
            x = x + self.pos_embed[:, :seq_len]

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Reshape back to spatial: [B, H*W, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class HybridFPN(nn.Module):
    """Feature Pyramid Network that fuses CNN multi-scale features with
    transformer-enhanced features."""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()

        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for ch in in_channels
        ])

        # Top-down refinement
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            for _ in in_channels
        ])

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:],
                                      mode='bilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Refinement
        outputs = [conv(lat) for conv, lat in zip(self.td_convs, laterals)]
        return outputs


class MultiScaleDetectionHead(nn.Module):
    """Multi-scale detection head with dilated convolutions and CBAM attention."""

    def __init__(self, in_channels: int = 256, hidden_channels: int = 128):
        super().__init__()

        # Fine branch (dilation=1)
        self.fine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        # Medium branch (dilation=2)
        self.medium = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        # Coarse branch (dilation=4)
        self.coarse = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)

        # Final output
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.scale_weights, dim=0)
        out = w[0] * self.fine(x) + w[1] * self.medium(x) + w[2] * self.coarse(x)
        return self.out_conv(out)


@MODELS.register()
class HerdNetHybrid(nn.Module):
    """HerdNet with ConvNeXt CNN stem + Transformer global reasoning.

    Architecture:
        ConvNeXt Stage 0: 512 -> 128 (1/4 res), preserves fine spatial detail
        ConvNeXt Stage 1: 128 -> 64  (1/8 res), 40-50px objects are 5-6px features
        Transformer Encoder: Global self-attention at 1/8 resolution (4096 tokens)
        ConvNeXt Stage 2: 64 -> 32   (1/16 res), deeper features with context
        ConvNeXt Stage 3: 32 -> 16   (1/32 res), coarsest features
        FPN: Multi-scale feature fusion
        Detection Head: Multi-scale heatmap prediction
        Classification Head: Per-class prediction at 1/32

    The transformer blocks sit between stage 1 and stage 2, operating on
    the 1/8 resolution features where small objects still have spatial extent
    but the sequence length (64x64=4096) is manageable.

    Args:
        backbone_size: ConvNeXt size ('tiny', 'small', 'base')
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained ConvNeXt weights
        down_ratio: Output heatmap downsampling ratio
        transformer_dim: Channel dimension for transformer (projected from stage 1 output)
        transformer_depth: Number of transformer blocks
        transformer_heads: Number of attention heads
        transformer_mlp_ratio: MLP expansion ratio in transformer
        transformer_drop: Dropout rate in transformer
        fpn_channels: FPN output channels
        debug: Print architecture info
    """

    def __init__(
        self,
        backbone_size: str = 'tiny',
        num_classes: int = 2,
        pretrained: bool = True,
        down_ratio: int = 4,
        transformer_dim: int = 256,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_drop: float = 0.1,
        fpn_channels: int = 256,
        debug: bool = True,
    ):
        super().__init__()

        assert backbone_size in ['tiny', 'small', 'base']
        assert down_ratio in [1, 2, 4, 8, 16]

        self.down_ratio = down_ratio
        self.num_classes = num_classes

        convnext_models = {
            'tiny': ('convnext_tiny.fb_in22k_ft_in1k', [96, 192, 384, 768]),
            'small': ('convnext_small.fb_in22k_ft_in1k', [96, 192, 384, 768]),
            'base': ('convnext_base.fb_in22k_ft_in1k', [128, 256, 512, 1024]),
        }
        model_name, self.feature_channels = convnext_models[backbone_size]

        if debug:
            logger.info(f"\nInitializing HerdNetHybrid:")
            logger.info(f"  CNN Stem: ConvNeXt-{backbone_size.capitalize()} (stages 0-3)")
            logger.info(f"  Transformer: {transformer_depth} blocks, dim={transformer_dim}, "
                        f"heads={transformer_heads} at 1/8 resolution")

        # Full ConvNeXt backbone (all 4 stages)
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except Exception:
            simple_name = model_name.split('.')[0]
            logger.warning(f"  Falling back to {simple_name}")
            self.backbone = timm.create_model(
                simple_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )

        # Project stage 1 features to transformer dim
        stage1_channels = self.feature_channels[1]  # 192 for tiny
        self.proj_to_transformer = nn.Sequential(
            nn.Conv2d(stage1_channels, transformer_dim, 1, bias=False),
            nn.BatchNorm2d(transformer_dim),
            nn.GELU(),
        )
        self.proj_from_transformer = nn.Sequential(
            nn.Conv2d(transformer_dim, stage1_channels, 1, bias=False),
            nn.BatchNorm2d(stage1_channels),
            nn.GELU(),
        )

        # Transformer encoder at 1/8 resolution
        self.transformer = SpatialTransformerEncoder(
            dim=transformer_dim,
            depth=transformer_depth,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            drop=transformer_drop,
            max_spatial=64,  # 512/8 = 64
        )

        if debug:
            logger.info(f"  Feature channels: {self.feature_channels}")
            logger.info(f"  Transformer input: {stage1_channels}ch -> {transformer_dim}ch")
            for i, (h, ch) in enumerate(zip([128, 64, 32, 16], self.feature_channels)):
                extra = " + Transformer" if i == 1 else ""
                logger.info(f"    Stage {i}: {ch} channels, {h}x{h}{extra}")

        # FPN takes all 4 stages
        self.fpn = HybridFPN(
            in_channels=self.feature_channels,
            out_channels=fpn_channels,
        )

        # Detection head on finest FPN scale
        self.detection_head = MultiScaleDetectionHead(
            in_channels=fpn_channels,
            hidden_channels=128,
        )

        # Classification head on coarsest scale
        self.cls_head = nn.Sequential(
            nn.Conv2d(fpn_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

        # Initialize cls head
        for m in self.cls_head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Temperature for calibrated predictions
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

        if debug:
            self.check_trainable_parameters()

    def check_trainable_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        transformer_total = sum(p.numel() for p in self.transformer.parameters())

        logger.info(f"  Total parameters: {total:,}")
        logger.info(f"  Trainable: {trainable:,} ({100 * trainable / total:.1f}%)")
        logger.info(f"  Backbone (CNN): {backbone_total:,}")
        logger.info(f"  Transformer: {transformer_total:,}")
        return {'total': total, 'trainable': trainable}

    def reshape_classes(self, num_classes: int) -> None:
        self.cls_head[-1] = nn.Conv2d(128, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.cls_head[-1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.cls_head[-1].bias)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            heatmap: [B, 1, H/down_ratio, W/down_ratio]
            classification: [B, num_classes, H/32, W/32]
        """
        original_h, original_w = x.shape[2], x.shape[3]

        # Extract all 4 CNN stages
        features = [f.contiguous() for f in self.backbone(x)]
        # features[0]: 1/4 res (128x128), features[1]: 1/8 res (64x64)
        # features[2]: 1/16 res (32x32), features[3]: 1/32 res (16x16)

        # Apply transformer at 1/8 resolution (stage 1 output)
        # This adds global context while preserving the CNN's spatial detail
        stage1_proj = self.proj_to_transformer(features[1])
        stage1_transformed = self.transformer(stage1_proj)
        # Residual connection: CNN features + transformer context
        features[1] = features[1] + self.proj_from_transformer(stage1_transformed)

        # FPN fusion
        fpn_features = self.fpn(features)

        # Detection heatmap from finest scale
        det_logits = self.detection_head(fpn_features[0])
        heatmap = torch.sigmoid(det_logits / self.temperature.clamp(min=0.1))

        # Resize to target
        target_heatmap_size = (original_h // self.down_ratio, original_w // self.down_ratio)
        if heatmap.shape[2:] != target_heatmap_size:
            heatmap = F.interpolate(heatmap, size=target_heatmap_size,
                                    mode='bilinear', align_corners=False)

        # Classification from coarsest scale
        cls_out = self.cls_head(fpn_features[-1])
        cls_target_size = (original_h // 32, original_w // 32)
        if cls_out.shape[2:] != cls_target_size:
            cls_out = F.interpolate(cls_out, size=cls_target_size,
                                    mode='bilinear', align_corners=False)

        return heatmap, cls_out
