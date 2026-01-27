import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import math


class DINOv2Encoder(nn.Module):
    """
    DINOv2 encoder that extracts multi-scale features for UNet.
    Uses the rich representations learned from 142M images.
    """

    def __init__(
            self,
            model_name: str = 'dinov2_vitb14',
            freeze_backbone: bool = False,
            output_layers: List[int] = [3, 6, 9, 12]  # Which transformer layers to extract
    ):
        super().__init__()

        # Load DINOv2 model (trained on 142M images)
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model_name = model_name
        self.output_layers = output_layers

        # Get model specifications
        self.patch_size = self.dinov2.patch_embed.patch_size[0]
        self.embed_dim = self.dinov2.embed_dim
        self.num_layers = len(self.dinov2.blocks)

        print(f"DINOv2 Encoder loaded: {model_name}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Total layers: {self.num_layers}")
        print(f"  Output layers: {output_layers}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            print("  Backbone frozen ❄️")

        # Hook to extract intermediate features
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to extract intermediate features."""

        def get_activation(name):
            def hook(module, input, output):
                self.features[name] = output

            return hook

        # Register hooks for specified layers
        for layer_idx in self.output_layers:
            if layer_idx < len(self.dinov2.blocks):
                hook = self.dinov2.blocks[layer_idx].register_forward_hook(
                    get_activation(f'layer_{layer_idx}')
                )
                self.hooks.append(hook)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from DINOv2.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Dictionary of feature maps at different scales
        """
        B, C, H, W = x.shape

        # Clear previous features
        self.features.clear()

        # Forward pass through DINOv2
        _ = self.dinov2.forward_features(x)

        # Convert patch tokens to spatial feature maps
        feature_maps = {}

        for layer_idx in self.output_layers:
            layer_name = f'layer_{layer_idx}'
            if layer_name in self.features:
                # Get patch tokens (excluding CLS token)
                tokens = self.features[layer_name]  # [B, N_patches + 1, D]
                patch_tokens = tokens[:, 1:]  # Remove CLS token [B, N_patches, D]

                # Calculate spatial dimensions
                n_patches = patch_tokens.shape[1]
                patch_h = patch_w = int(math.sqrt(n_patches))

                # Reshape to spatial feature map
                feature_map = patch_tokens.transpose(1, 2).reshape(
                    B, self.embed_dim, patch_h, patch_w
                )  # [B, D, H_patch, W_patch]

                feature_maps[f'stage_{layer_idx}'] = feature_map

        return feature_maps

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class UNetDecoder(nn.Module):
    """
    UNet decoder with skip connections for DINOv2 features.
    """

    def __init__(
            self,
            encoder_channels: List[int],
            decoder_channels: List[int] = [512, 256, 128, 64],
            num_classes: int = 21,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.encoder_channels = encoder_channels  # From deepest to shallowest
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes

        # Build decoder blocks
        self.decoder_blocks = nn.ModuleList()

        # First decoder block (no skip connection)
        self.decoder_blocks.append(
            DecoderBlock(
                in_channels=encoder_channels[0],
                out_channels=decoder_channels[0],
                dropout_rate=dropout_rate
            )
        )

        # Subsequent decoder blocks with skip connections
        for i in range(1, len(decoder_channels)):
            # Skip connection adds encoder features
            skip_channels = encoder_channels[i] if i < len(encoder_channels) else 0
            in_channels = decoder_channels[i - 1] + skip_channels
            out_channels = decoder_channels[i]

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate
                )
            )

        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        )

        print(f"UNet Decoder created:")
        print(f"  Encoder channels: {encoder_channels}")
        print(f"  Decoder channels: {decoder_channels}")
        print(f"  Number of classes: {num_classes}")

    def forward(
            self,
            encoder_features: Dict[str, torch.Tensor],
            target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Decode features with skip connections.

        Args:
            encoder_features: Dictionary of multi-scale features from encoder
            target_size: Target output size (H, W)

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Get features in order (deepest first)
        feature_keys = sorted(encoder_features.keys(), reverse=True)
        features = [encoder_features[key] for key in feature_keys]

        # Start with deepest features
        x = features[0]
        x = self.decoder_blocks[0](x)

        # Apply subsequent decoder blocks with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks[1:], 1):
            # Upsample current features
            if i < len(features):
                skip_features = features[i]
                x = F.interpolate(
                    x,
                    size=skip_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                # Concatenate skip connection
                x = torch.cat([x, skip_features], dim=1)
            else:
                # Upsample without skip connection
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            # Apply decoder block
            x = decoder_block(x)

        # Apply segmentation head
        x = self.segmentation_head(x)

        # Final upsample to target size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x


class DecoderBlock(nn.Module):
    """Individual decoder block with double convolution."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DINOv2UNet(nn.Module):
    """
    Complete DINOv2-UNet model for semantic segmentation.
    Combines DINOv2 encoder (142M images pretrained) with UNet decoder.
    """

    def __init__(
            self,
            model_name: str = 'dinov2_vitb14',
            num_classes: int = 21,
            decoder_channels: List[int] = [512, 256, 128, 64],
            freeze_encoder: bool = False,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.num_classes = num_classes

        # Create encoder
        self.encoder = DINOv2Encoder(
            model_name=model_name,
            freeze_backbone=freeze_encoder
        )

        # Get encoder channel dimensions (all layers have same dim in DINOv2)
        encoder_channels = [self.encoder.embed_dim] * len(self.encoder.output_layers)

        # Create decoder
        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

        print(f"\n🎯 DINOv2-UNet Model Summary:")
        print(f"  Architecture: {model_name} + UNet Decoder")
        print(f"  Pretraining: 142M images (self-supervised)")
        print(f"  Classes: {num_classes}")
        print(f"  Encoder frozen: {freeze_encoder}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder-decoder."""
        B, C, H, W = x.shape

        # Extract multi-scale features from DINOv2
        encoder_features = self.encoder(x)

        # Decode with skip connections
        output = self.decoder(encoder_features, target_size=(H, W))

        return output

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen ❄️")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen 🔥")


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with DINOv2-UNet."""

    def __init__(
            self,
            images_dir: Union[str, Path],
            masks_dir: Union[str, Path],
            img_size: int = 518,  # DINOv2 was trained up to 518x518
            num_classes: int = 21,
            augmentations: Optional[A.Compose] = None,
            is_training: bool = True
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.num_classes = num_classes
        self.augmentations = augmentations
        self.is_training = is_training

        # Get image files
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) +
                                  list(self.images_dir.glob('*.png')) +
                                  list(self.images_dir.glob('*.jpeg')))

        print(f"Found {len(self.image_files)} images in {images_dir}")

        # Basic transforms
        self.basic_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.masks_dir / (img_path.stem + '.png')
        if not mask_path.exists():
            mask_path = self.masks_dir / (img_path.stem + '.jpg')

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Ensure mask values are valid
        mask = np.clip(mask, 0, self.num_classes - 1)

        # Apply augmentations
        if self.augmentations and self.is_training:
            augmented = self.augmentations(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        else:
            transformed = self.basic_transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, mask.long()


class DINOv2UNetTrainer:
    """Training pipeline for DINOv2-UNet."""

    def __init__(
            self,
            model: DINOv2UNet,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str = 'cpu',
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            use_class_weights: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        if use_class_weights:
            # You can compute class weights from your dataset
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Optimizer - different learning rates for encoder vs decoder
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained encoder
            {'params': decoder_params, 'lr': learning_rate}  # Higher LR for decoder
        ], weight_decay=weight_decay)

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': []
        }

        print(f"Trainer initialized on {device}")
        print(f"Encoder LR: {learning_rate * 0.1}")
        print(f"Decoder LR: {learning_rate}")

    def calculate_iou(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate mean IoU."""
        predictions = torch.argmax(predictions, dim=1)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Remove ignored pixels
        valid_mask = targets != 255
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if len(predictions) == 0:
            return 0.0

        # Calculate IoU for each class
        num_classes = self.model.num_classes
        ious = []

        for cls in range(num_classes):
            pred_cls = predictions == cls
            target_cls = targets == cls

            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()

            if union > 0:
                ious.append(intersection / union)

        return np.mean(ious) if ious else 0.0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_iou = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            total_loss += loss.item()
            iou = self.calculate_iou(outputs.detach(), masks)
            total_iou += iou
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })

        return {
            'loss': total_loss / num_batches,
            'iou': total_iou / num_batches
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Calculate metrics
                total_loss += loss.item()
                iou = self.calculate_iou(outputs, masks)
                total_iou += iou
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}'
                })

        return {
            'loss': total_loss / num_batches,
            'iou': total_iou / num_batches
        }

    def train(self, num_epochs: int, save_dir: str = 'checkpoints'):
        """Complete training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        best_iou = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train and validate
            train_results = self.train_epoch()
            val_results = self.validate_epoch()

            # Update scheduler
            self.scheduler.step()

            # Update history
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['train_iou'].append(train_results['iou'])
            self.history['val_iou'].append(val_results['iou'])

            # Print results
            print(f"Train - Loss: {train_results['loss']:.4f}, IoU: {train_results['iou']:.4f}")
            print(f"Val   - Loss: {val_results['loss']:.4f}, IoU: {val_results['iou']:.4f}")

            # Save best model
            if val_results['iou'] > best_iou:
                best_iou = val_results['iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_iou': best_iou,
                    'history': self.history
                }, save_dir / 'best_model.pth')
                print(f"New best model saved! IoU: {best_iou:.4f}")

        return self.history


def create_augmentations():
    """Create augmentation pipelines."""
    train_transform = A.Compose([
        A.Resize(518, 518),  # DINOv2 trained up to 518x518
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(518, 518),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


def visualize_predictions(
        model: DINOv2UNet,
        dataset: SegmentationDataset,
        device: str = 'cpu',
        num_samples: int = 4
):
    """Visualize model predictions."""
    model.eval()

    fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
    fig.suptitle('DINOv2-UNet Segmentation Results', fontsize=16)

    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            image, mask = dataset[i]
            image_tensor = image.unsqueeze(0).to(device)

            # Predict
            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

            # Convert image for visualization
            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
                      np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np, 0, 1)

            # Plot
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f'Image {i + 1}')
            axes[0, i].axis('off')

            axes[1, i].imshow(mask.numpy(), cmap='tab20')
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')

            axes[2, i].imshow(pred_mask, cmap='tab20')
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')

    plt.tight_layout()
    return fig


def demo_dinov2_unet():
    """Demonstrate DINOv2-UNet pipeline."""
    print("🚀 DINOv2-UNet Semantic Segmentation Pipeline")
    print("=" * 60)

    # Model variants comparison
    model_variants = {
        'dinov2_vits14': {'params': '21M', 'speed': 'Fast', 'memory': 'Low'},
        'dinov2_vitb14': {'params': '86M', 'speed': 'Medium', 'memory': 'Medium'},
        'dinov2_vitl14': {'params': '300M', 'speed': 'Slow', 'memory': 'High'},
        'dinov2_vitg14': {'params': '1.1B', 'speed': 'Very Slow', 'memory': 'Very High'},
    }

    print("\n📊 Available DINOv2 Models:")
    print("-" * 40)
    for model, specs in model_variants.items():
        print(f"{model:<20} | {specs['params']:<8} | {specs['speed']:<10} | {specs['memory']}")

    # Create model
    print(f"\n🏗️ Creating DINOv2-UNet model...")
    model = DINOv2UNet(
        model_name='dinov2_vitb14',  # Good balance of performance/speed
        num_classes=21,
        freeze_encoder=True  # Start with frozen encoder
    )

    # Test forward pass
    dummy_input = torch.randn(2, 3, 518, 518)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✅ Model test successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📈 Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    return model


if __name__ == "__main__":
    # Run demonstration
    model = demo_dinov2_unet()

    print("\n" + "=" * 70)
    print("🎯 USAGE EXAMPLE:")
    print("=" * 70)
    print("""
# 1. Create model
model = DINOv2UNet(
    model_name='dinov2_vitb14',  # or 'dinov2_vitl14' for better quality
    num_classes=21,
    freeze_encoder=True  # Fine-tune decoder first
)

# 2. Create datasets
train_transform, val_transform = create_augmentations()

train_dataset = SegmentationDataset(
    images_dir='path/to/train/images',
    masks_dir='path/to/train/masks',
    img_size=518,  # DINOv2 optimal size
    augmentations=train_transform
)

val_dataset = SegmentationDataset(
    images_dir='path/to/val/images',
    masks_dir='path/to/val/masks',
    img_size=518,
    augmentations=val_transform,
    is_training=False
)

# 3. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 4. Train model
trainer = DINOv2UNetTrainer(model, train_loader, val_loader)

# Phase 1: Train decoder only (encoder frozen)
history1 = trainer.train(num_epochs=20)

# Phase 2: Fine-tune entire model
model.unfreeze_encoder()
trainer.optimizer = optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 1e-5},  # Very low LR
    {'params': model.decoder.parameters(), 'lr': 1e-4}   # Higher LR
], weight_decay=1e-4)

history2 = trainer.train(num_epochs=10)

# 5. Visualize results
visualize_predictions(model, val_dataset)
    """)

    print("\n🌟 KEY ADVANTAGES:")
    print("✅ DINOv2 trained on 142M images (100x more than ImageNet)")
    print("✅ Self-supervised features - no labels needed for pretraining")
    print("✅ Handles up to 518x518 images efficiently")
    print("✅ Rich multi-scale features perfect for segmentation")
    print("✅ Two-phase training: freeze encoder → fine-tune all")
    print("✅ Skip connections preserve fine details")

    print("\n💡 TRAINING TIPS:")
    print("• Start with frozen encoder for stable training")
    print("• Use different learning rates for encoder vs decoder")
    print("• DINOv2 works best at 518x518 resolution")
    print("• Consider data augmentation carefully - DINOv2 is robust")
    print("• Fine-tune end-to-end after decoder converges")