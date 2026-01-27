import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import timm
from typing import List, Optional, Union

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

from typing import List
import torch

import torch.nn as nn
import numpy as np
import torchvision.transforms as T

from typing import Optional

from .register import MODELS

from . import dla as dla_modules
import timm
BatchNorm = nn.BatchNorm2d

BatchNorm = nn.BatchNorm2d

@MODELS.register()
class HerdNetTimmBackbone(nn.Module):
    """HerdNet architecture with ConvNeXt or DiNAv2 backbone from timm"""

    def __init__(
            self,
            backbone_name: str = 'convnext_tiny',
            num_classes: int = 2,
            pretrained: bool = True,
            down_ratio: int = 2,
            head_conv: int = 64,
            backbone_type: str = 'convnext'  # 'convnext' or 'dinav2'
    ):
        """
        Args:
            backbone_name (str): Name of the backbone model (e.g., 'convnext_tiny', 'vit_base_patch14_dinov2')
            num_classes (int): Number of output classes, background included. Defaults to 2.
            pretrained (bool): Whether to use pretrained weights. Defaults to True.
            down_ratio (int): Downsample ratio. Possible values are 1, 2, 4, 8, or 16. Defaults to 2.
            head_conv (int): Number of channels in head convolutions. Defaults to 64.
            backbone_type (str): Type of backbone - 'convnext' or 'dinav2'
        """
        super(HerdNetTimmBackbone, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.backbone_type = backbone_type
        self.first_level = int(np.log2(down_ratio))

        # Initialize backbone
        if backbone_type == 'convnext':
            self._init_convnext_backbone(backbone_name, pretrained)
        elif backbone_type == 'dinav2':
            self._init_dinav2_backbone(backbone_name, pretrained)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}. Use 'convnext' or 'dinav2'")

        # Get channels for the levels we'll use
        channels_for_upsampling = self.channels[self.first_level:]

        # Bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            self.channels[-1], self.channels[-1],
            kernel_size=1, stride=1,
            padding=0, bias=True
        )

        # Create FPN-style components
        self.lateral_connections = self._make_lateral_connections(channels_for_upsampling)
        self.upsampling_layers = self._make_upsampling_layers(channels_for_upsampling)
        self.fusion_nodes = self._make_fusion_nodes(channels_for_upsampling)

        # Final projection layer to match expected channels for localization head
        self.final_proj = nn.Conv2d(
            self.channels[-1], self.channels[self.first_level],
            kernel_size=1, stride=1, padding=0, bias=False
        )

        # Build heads
        # self._build_localization_head()
        self._build_heatmap_head()
        self._build_classification_head()

        # Initialize bias
        self.cls_head[-1].bias.data.fill_(0.00)

    def _init_convnext_backbone(self, backbone_name: str,
                                pretrained: bool = True,
                                load_from = None):
        """Initialize ConvNeXt backbone"""
        # Create ConvNeXt model with feature extraction (ConvNeXt has 4 stages by default)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True
        )

        # Get feature info to determine channels
        feature_info = self.backbone.feature_info
        backbone_channels = [info['num_chs'] for info in feature_info]

        # ConvNeXt typically has 4 stages, but we need 5 for compatibility
        # Add a stem stage at the beginning with appropriate channels
        if 'tiny' in backbone_name:
            stem_channels = 96
        elif 'small' in backbone_name:
            stem_channels = 128
        elif 'base' in backbone_name:
            stem_channels = 128
        elif 'large' in backbone_name:
            stem_channels = 192
        else:
            stem_channels = backbone_channels[0] // 2

        # Create 5-stage channel list: [stem, stage1, stage2, stage3, stage4]
        self.channels = [stem_channels] + backbone_channels

        # Add a stem projection to create the first feature level
        self.stem_proj = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=4, stride=4, padding=0),
            nn.GroupNorm(1, stem_channels),  # LayerNorm equivalent for 2D
            nn.GELU()
        )


    def _make_lateral_connections(self, channels: List[int]):
        """Create lateral connections for FPN-style upsampling"""
        layers = nn.ModuleList()

        for i in range(len(channels) - 1):
            lateral = nn.Sequential(
                nn.Conv2d(channels[i], channels[-1], kernel_size=1, stride=1, padding=0, bias=False),
                BatchNorm(channels[-1]),
                nn.ReLU(inplace=True)
            )
            layers.append(lateral)

        return layers

    def _make_upsampling_layers(self, channels: List[int]):
        """Create upsampling layers"""
        layers = nn.ModuleList()
        scales = [2 ** i for i in range(len(channels) - 1)]

        for i in range(len(channels) - 1):
            factor = scales[i]

            if factor == 1:
                up_layer = nn.Identity()
            else:
                up_layer = nn.ConvTranspose2d(
                    channels[-1], channels[-1],
                    kernel_size=factor * 2,
                    stride=factor,
                    padding=factor // 2,
                    output_padding=0,
                    groups=channels[-1],
                    bias=False
                )
                self._fill_up_weights(up_layer)

            layers.append(up_layer)

        return layers

    def _make_fusion_nodes(self, channels: List[int]):
        """Create fusion nodes for feature aggregation"""
        nodes = nn.ModuleList()

        for i in range(len(channels) - 1):
            node = nn.Sequential(
                nn.Conv2d(channels[-1] * 2, channels[-1],
                          kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(channels[-1]),
                nn.ReLU(inplace=True)
            )
            nodes.append(node)

        return nodes

    def _build_localization_head(self):
        """Build the localization head with proper upsampling"""
        layers = []

        additional_upscale = 2 ** self.first_level

        if additional_upscale > 1:
            layers.append(nn.ConvTranspose2d(
                self.channels[self.first_level], self.channels[self.first_level],
                kernel_size=additional_upscale * 2,
                stride=additional_upscale,
                padding=additional_upscale // 2,
                output_padding=0,
                groups=self.channels[self.first_level],
                bias=False
            ))
            self._fill_up_weights(layers[-1])

        layers.extend([
            nn.Conv2d(self.channels[self.first_level], self.head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.head_conv, 1,
                kernel_size=1, stride=1,
                padding=0, bias=True
            ),
            nn.Sigmoid()
        ])

        self.loc_head = nn.Sequential(*layers)
        self.loc_head[-2].bias.data.fill_(0.00)

    def _build_heatmap_head(self):
        """Build the heatmap head"""
        self.hm_head = nn.Sequential(
            nn.Conv2d(self.channels[self.first_level], self.head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.head_conv, self.num_classes - 1,
                kernel_size=1, stride=1,
                padding=0, bias=True
            ),
            nn.Sigmoid()
        )

    def _build_classification_head(self):
        """Build the classification head"""
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.channels[-1], self.head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.head_conv, self.num_classes,
                kernel_size=1, stride=1,
                padding=0, bias=True
            )
        )

    def _fill_up_weights(self, up):
        """Initialize upsampling weights for bilinear interpolation"""
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, input: torch.Tensor):
        """Forward pass"""
        # Extract features from backbone
        if self.backbone_type == 'convnext':
            # Get features from ConvNeXt backbone (4 stages)
            backbone_features = self.backbone(input)

            # Add stem feature as first level
            stem_feature = self.stem_proj(input)
            features = [stem_feature] + list(backbone_features)

        elif self.backbone_type == 'dinav2':
            backbone_features = self.backbone(input)

            if hasattr(self, 'feature_pyramid'):
                # Single feature case - create pyramid
                base_feature = backbone_features[-1]  # Use the last (and likely only) feature
                features = []
                for i, proj in enumerate(self.feature_pyramid):
                    # Apply different downsampling for each level
                    if i == 0:  # Most downsampled
                        feat = F.adaptive_avg_pool2d(base_feature,
                                                     (base_feature.shape[2] // 8, base_feature.shape[3] // 8))
                    elif i == 1:
                        feat = F.adaptive_avg_pool2d(base_feature,
                                                     (base_feature.shape[2] // 4, base_feature.shape[3] // 4))
                    elif i == 2:
                        feat = F.adaptive_avg_pool2d(base_feature,
                                                     (base_feature.shape[2] // 2, base_feature.shape[3] // 2))
                    else:
                        feat = base_feature

                    features.append(proj(feat))
            else:
                # Multi-scale case
                features = list(backbone_features)
                # Ensure we have 5 features
                while len(features) < 5:
                    features = [features[0]] + features

        else:
            features = self.backbone(input)
            # Ensure we have the right number of features
            while len(features) < 5:
                features = [features[0]] + features

        # Get the features we need for upsampling based on first_level
        features_for_upsampling = features[self.first_level:]

        # Apply bottleneck to the deepest feature
        bottleneck = self.bottleneck_conv(features_for_upsampling[-1])
        features_for_upsampling[-1] = bottleneck

        # Apply lateral connections to all features except the deepest
        laterals = []
        for i, feature in enumerate(features_for_upsampling[:-1]):
            lateral = self.lateral_connections[i](feature)
            laterals.append(lateral)

        # Add the bottleneck feature as the last one
        laterals.append(bottleneck)

        # Start the top-down pathway from the deepest feature
        x = laterals[-1]

        # Process features from deepest to shallowest
        for i in range(len(laterals) - 2, -1, -1):
            # Upscale the current feature
            upsampled = self.upsampling_layers[i](x)

            # Ensure spatial dimensions match
            if upsampled.shape[2:] != laterals[i].shape[2:]:
                upsampled = F.interpolate(
                    upsampled,
                    size=laterals[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Merge features
            x = self.fusion_nodes[i](torch.cat([upsampled, laterals[i]], dim=1))

        # Final feature map for localization
        decode_hm = x

        # Transform channel dimension for localization head
        decode_hm = self.final_proj(decode_hm)

        # Generate outputs
        heatmap = self.loc_head(decode_hm)
        clsmap = self.cls_head(bottleneck)
        real_heatmap = self.hm_head(decode_hm)

        return heatmap, clsmap, real_heatmap

    def freeze(self, layers: list) -> None:
        """Freeze all layers mentioned in the input list"""
        for layer in layers:
            self._freeze_layer(layer)

    def _freeze_layer(self, layer_name: str) -> None:
        """Freeze a specific layer"""
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    def reshape_classes(self, num_classes: int) -> None:
        """Reshape architecture according to a new number of classes"""
        self.cls_head[-1] = nn.Conv2d(
            self.head_conv, num_classes,
            kernel_size=1, stride=1,
            padding=0, bias=True
        )

        self.hm_head[-2] = nn.Conv2d(
            self.head_conv, num_classes - 1,
            kernel_size=1, stride=1,
            padding=0, bias=True
        )

        self.cls_head[-1].bias.data.fill_(0.00)
        self.num_classes = num_classes


# Factory functions for easy model creation
def create_herdnet_convnext(
        model_size: str = 'tiny',
        num_classes: int = 2,
        pretrained: bool = True,
        down_ratio: int = 2,
        head_conv: int = 64
) -> HerdNetTimmBackbone:
    """
    Create HerdNet with ConvNeXt backbone

    Args:
        model_size: 'tiny', 'small', 'base', 'large'
        num_classes: Number of classes
        pretrained: Use pretrained weights
        down_ratio: Downsample ratio
        head_conv: Head convolution channels
    """
    backbone_name = f'convnext_{model_size}'
    return HerdNetTimmBackbone(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        down_ratio=down_ratio,
        head_conv=head_conv,
        backbone_type='convnext'
    )


def create_herdnet_dinav2(
        model_size: str = 'base',
        patch_size: int = 14,
        num_classes: int = 2,
        pretrained: bool = True,
        down_ratio: int = 2,
        head_conv: int = 64
) -> HerdNetTimmBackbone:
    """
    Create HerdNet with DiNAv2 backbone

    Args:
        model_size: 'small', 'base', 'large', 'giant'
        patch_size: Patch size (14 or 16)
        num_classes: Number of classes
        pretrained: Use pretrained weights
        down_ratio: Downsample ratio
        head_conv: Head convolution channels
    """
    backbone_name = f'vit_{model_size}_patch{patch_size}_dinov2'
    return HerdNetTimmBackbone(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        down_ratio=down_ratio,
        head_conv=head_conv,
        backbone_type='dinav2'
    )


# Example usage:
if __name__ == "__main__":
    # Create ConvNeXt-based HerdNet
    model_convnext = create_herdnet_convnext(
        model_size='tiny',
        num_classes=3,
        down_ratio=2
    )


    # Test forward pass
    x = torch.randn(2, 3, 512, 512)

    print("ConvNeXt Model:")
    heatmap, clsmap, real_heatmap = model_convnext(x)
    print(f"Input shape: {x.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Classification map shape: {clsmap.shape}")
    print(f"Real heatmap shape: {real_heatmap.shape}")

