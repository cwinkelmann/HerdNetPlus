import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import timm
from typing import List, Optional, Union

BatchNorm = nn.BatchNorm2d


class HerdNetSegmentationBackbone(nn.Module):
    """HerdNet-style architecture adapted for semantic segmentation with ConvNeXt or DiNAv2 backbone"""

    def __init__(
            self,
            backbone_name: str = 'convnext_tiny',
            num_classes: int = 21,  # Pascal VOC default
            pretrained: bool = True,
            down_ratio: int = 4,
            head_conv: int = 128,
            backbone_type: str = 'convnext',  # 'convnext' or 'dinav2'
            use_auxiliary_head: bool = True,
            dropout_rate: float = 0.1
    ):
        """
        Args:
            backbone_name (str): Name of the backbone model
            num_classes (int): Number of segmentation classes (including background)
            pretrained (bool): Whether to use pretrained weights
            down_ratio (int): Downsample ratio for main segmentation head
            head_conv (int): Number of channels in segmentation heads
            backbone_type (str): Type of backbone - 'convnext' or 'dinav2'
            use_auxiliary_head (bool): Whether to use auxiliary segmentation head
            dropout_rate (float): Dropout rate in segmentation heads
        """
        super(HerdNetSegmentationBackbone, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.backbone_type = backbone_type
        self.use_auxiliary_head = use_auxiliary_head
        self.dropout_rate = dropout_rate
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

        # Multi-scale feature processing
        self.feature_refinement = self._make_feature_refinement(channels_for_upsampling)

        # ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context
        self.aspp = ASPP(self.channels[-1], 256, [6, 12, 18])

        # FPN-style components
        self.lateral_connections = self._make_lateral_connections(channels_for_upsampling)
        self.upsampling_layers = self._make_upsampling_layers(channels_for_upsampling)
        self.fusion_nodes = self._make_fusion_nodes(channels_for_upsampling)

        # Final projection layer
        self.final_proj = nn.Conv2d(
            self.channels[-1], self.channels[self.first_level],
            kernel_size=1, stride=1, padding=0, bias=False
        )

        # Build segmentation heads
        self._build_main_segmentation_head()

        if self.use_auxiliary_head:
            self._build_auxiliary_segmentation_head()

        # Initialize weights
        self._initialize_weights()

    def _init_convnext_backbone(self, backbone_name: str, pretrained: bool):
        """Initialize ConvNeXt backbone"""
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True
        )

        feature_info = self.backbone.feature_info
        backbone_channels = [info['num_chs'] for info in feature_info]

        # Add stem stage
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

        self.channels = [stem_channels] + backbone_channels

        self.stem_proj = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=4, stride=4, padding=0),
            nn.GroupNorm(1, stem_channels),
            nn.GELU()
        )

    def _init_dinav2_backbone(self, backbone_name: str, pretrained: bool):
        """Initialize DiNAv2 backbone"""
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True
        )

        feature_info = self.backbone.feature_info
        backbone_channels = [info['num_chs'] for info in feature_info]

        if len(backbone_channels) == 1:
            base_dim = backbone_channels[0]
            self.channels = [
                base_dim // 4,
                base_dim // 2,
                base_dim,
                base_dim,
                base_dim
            ]

            self.feature_pyramid = nn.ModuleList([
                nn.Conv2d(base_dim, self.channels[i], 1, bias=False)
                for i in range(5)
            ])
        else:
            while len(backbone_channels) < 5:
                backbone_channels = [backbone_channels[0] // 2] + backbone_channels
            self.channels = backbone_channels[:5]

    def _make_feature_refinement(self, channels: List[int]):
        """Create feature refinement modules"""
        refinement = nn.ModuleList()

        for ch in channels:
            refine = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(ch),
                nn.ReLU(inplace=True)
            )
            refinement.append(refine)

        return refinement

    def _make_lateral_connections(self, channels: List[int]):
        """Create lateral connections for FPN-style upsampling"""
        layers = nn.ModuleList()

        for i in range(len(channels) - 1):
            lateral = nn.Sequential(
                nn.Conv2d(channels[i], 256, kernel_size=1, stride=1, padding=0, bias=False),
                BatchNorm(256),
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
                up_layer = nn.Sequential(
                    nn.ConvTranspose2d(
                        256, 256,
                        kernel_size=factor * 2,
                        stride=factor,
                        padding=factor // 2,
                        output_padding=0,
                        bias=False
                    ),
                    BatchNorm(256),
                    nn.ReLU(inplace=True)
                )
                self._fill_up_weights(up_layer[0])

            layers.append(up_layer)

        return layers

    def _make_fusion_nodes(self, channels: List[int]):
        """Create fusion nodes for feature aggregation"""
        nodes = nn.ModuleList()

        for i in range(len(channels) - 1):
            node = nn.Sequential(
                nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.dropout_rate)
            )
            nodes.append(node)

        return nodes

    def _build_main_segmentation_head(self):
        """Build the main segmentation head"""
        layers = []

        # Upsample to original resolution
        additional_upscale = 2 ** self.first_level
        if additional_upscale > 1:
            layers.extend([
                nn.ConvTranspose2d(
                    self.channels[self.first_level], self.head_conv,
                    kernel_size=additional_upscale * 2,
                    stride=additional_upscale,
                    padding=additional_upscale // 2,
                    output_padding=0,
                    bias=False
                ),
                BatchNorm(self.head_conv),
                nn.ReLU(inplace=True)
            ])
            self._fill_up_weights(layers[0])
        else:
            layers.extend([
                nn.Conv2d(self.channels[self.first_level], self.head_conv,
                          kernel_size=3, padding=1, bias=False),
                BatchNorm(self.head_conv),
                nn.ReLU(inplace=True)
            ])

        # Segmentation head
        layers.extend([
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(self.head_conv, self.head_conv, kernel_size=3, padding=1, bias=False),
            BatchNorm(self.head_conv),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(self.head_conv, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        ])

        self.segmentation_head = nn.Sequential(*layers)

    def _build_auxiliary_segmentation_head(self):
        """Build auxiliary segmentation head for training stability"""
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.channels[-2], self.head_conv, kernel_size=3, padding=1, bias=False),
            BatchNorm(self.head_conv),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(self.head_conv, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
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

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input: torch.Tensor):
        """Forward pass"""
        original_size = input.shape[2:]

        # Extract features from backbone
        if self.backbone_type == 'convnext':
            backbone_features = self.backbone(input)
            stem_feature = self.stem_proj(input)
            features = [stem_feature] + list(backbone_features)


        else:
            features = self.backbone(input)
            while len(features) < 5:
                features = [features[0]] + features

        # Get features for upsampling
        features_for_upsampling = features[self.first_level:]

        # Apply feature refinement
        for i, feature in enumerate(features_for_upsampling):
            features_for_upsampling[i] = self.feature_refinement[i](feature)

        # Apply ASPP to the deepest feature
        aspp_feature = self.aspp(features_for_upsampling[-1])

        # Apply lateral connections
        laterals = []
        for i, feature in enumerate(features_for_upsampling[:-1]):
            lateral = self.lateral_connections[i](feature)
            laterals.append(lateral)

        # Add ASPP feature as the deepest lateral
        laterals.append(aspp_feature)

        # Top-down pathway
        x = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = self.upsampling_layers[i](x)

            if upsampled.shape[2:] != laterals[i].shape[2:]:
                upsampled = F.interpolate(
                    upsampled,
                    size=laterals[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            x = self.fusion_nodes[i](torch.cat([upsampled, laterals[i]], dim=1))

        # Final feature processing
        decode_features = self.final_proj(x)

        # Main segmentation output
        main_output = self.segmentation_head(decode_features)

        # Upsample to original size
        main_output = F.interpolate(
            main_output,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )

        # Auxiliary output if enabled
        aux_output = None
        if self.use_auxiliary_head and self.training:
            aux_features = features_for_upsampling[-2]  # Use second-to-last feature
            aux_output = self.aux_head(aux_features)
            aux_output = F.interpolate(
                aux_output,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )

        if self.training and aux_output is not None:
            return main_output, aux_output
        else:
            return main_output

    def freeze(self, layers: list) -> None:
        """Freeze specified layers"""
        for layer in layers:
            self._freeze_layer(layer)

    def _freeze_layer(self, layer_name: str) -> None:
        """Freeze a specific layer"""
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

    def reshape_classes(self, num_classes: int) -> None:
        """Reshape architecture for new number of classes"""
        # Update main segmentation head
        self.segmentation_head[-1] = nn.Conv2d(
            self.head_conv, num_classes,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Update auxiliary head if it exists
        if self.use_auxiliary_head:
            self.aux_head[-1] = nn.Conv2d(
                self.head_conv, num_classes,
                kernel_size=1, stride=1, padding=0, bias=True
            )

        self.num_classes = num_classes


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""

    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int]):
        super(ASPP, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                BatchNorm(out_channels),
                nn.ReLU(inplace=True)
            )
            self.atrous_convs.append(conv)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        features = []

        # 1x1 conv
        features.append(self.conv1x1(x))

        # Atrous convs
        for atrous_conv in self.atrous_convs:
            features.append(atrous_conv(x))

        # Global average pooling
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        features.append(global_feat)

        # Concatenate and project
        x = torch.cat(features, dim=1)
        return self.project(x)


# Factory functions
def create_herdnet_segmentation_convnext(
        model_size: str = 'tiny',
        num_classes: int = 21,
        pretrained: bool = True,
        down_ratio: int = 4,
        head_conv: int = 128,
        use_auxiliary_head: bool = True,
        dropout_rate: float = 0.1
) -> HerdNetSegmentationBackbone:
    """Create HerdNet segmentation model with ConvNeXt backbone"""
    backbone_name = f'convnext_{model_size}'
    return HerdNetSegmentationBackbone(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        down_ratio=down_ratio,
        head_conv=head_conv,
        backbone_type='convnext',
        use_auxiliary_head=use_auxiliary_head,
        dropout_rate=dropout_rate
    )


def create_herdnet_segmentation_dinav2(
        model_size: str = 'base',
        patch_size: int = 14,
        num_classes: int = 21,
        pretrained: bool = True,
        down_ratio: int = 4,
        head_conv: int = 128,
        use_auxiliary_head: bool = True,
        dropout_rate: float = 0.1
) -> HerdNetSegmentationBackbone:
    """Create HerdNet segmentation model with DiNAv2 backbone"""
    backbone_name = f'vit_{model_size}_patch{patch_size}_dinov2'
    return HerdNetSegmentationBackbone(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        down_ratio=down_ratio,
        head_conv=head_conv,
        backbone_type='dinav2',
        use_auxiliary_head=use_auxiliary_head,
        dropout_rate=dropout_rate
    )


# Segmentation Loss Function
class SegmentationLoss(nn.Module):
    """Combined loss for semantic segmentation with auxiliary head"""

    def __init__(self, num_classes: int, ignore_index: int = 255, aux_weight: float = 0.4):
        super(SegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

        # Use label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.1)

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_output, aux_output = outputs
            main_loss = self.criterion(main_output, targets)
            aux_loss = self.criterion(aux_output, targets)
            return main_loss + self.aux_weight * aux_loss
        else:
            return self.criterion(outputs, targets)


# Example usage and testing
if __name__ == "__main__":
    # Create segmentation models
    print("Creating HerdNet Segmentation Models...")

    # ConvNeXt-based model
    model_convnext = create_herdnet_segmentation_convnext(
        model_size='tiny',
        num_classes=21,  # Pascal VOC
        down_ratio=4,
        use_auxiliary_head=True
    )



    # Test forward pass
    x = torch.randn(2, 3, 512, 512)

    print("\nConvNeXt Segmentation Model:")
    model_convnext.train()
    outputs_convnext = model_convnext(x)
    if isinstance(outputs_convnext, tuple):
        main_out, aux_out = outputs_convnext
        print(f"Input shape: {x.shape}")
        print(f"Main output shape: {main_out.shape}")
        print(f"Auxiliary output shape: {aux_out.shape}")
    else:
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {outputs_convnext.shape}")

    print(f"\nModel parameters: {sum(p.numel() for p in model_convnext.parameters()) / 1e6:.2f}M")

    # Test loss function
    criterion = SegmentationLoss(num_classes=21, aux_weight=0.4)
    targets = torch.randint(0, 21, (2, 512, 512))

    loss = criterion(outputs_convnext, targets)
    print(f"Training loss: {loss.item():.4f}")

    # Test inference mode
    model_convnext.eval()
    with torch.no_grad():
        inference_output = model_convnext(x)
        print(f"Inference output shape: {inference_output.shape}")

        # Get predictions
        predictions = torch.argmax(inference_output, dim=1)
        print(f"Predictions shape: {predictions.shape}")