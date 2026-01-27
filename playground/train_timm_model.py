import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import timm
import numpy as np
from tqdm import tqdm
import os
import logging
from pathlib import Path
import json
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimmModelTrainer:
    """Complete training pipeline for TIMM models"""

    def __init__(
            self,
            model_name: str = 'convnext_tiny',
            num_classes: int = 2,
            img_size: int = 224,
            pretrained: bool = True,
            device: str = 'auto'
    ):
        """
        Initialize trainer

        Args:
            model_name: TIMM model name (e.g., 'convnext_tiny', 'efficientnet_b0')
            num_classes: Number of output classes
            img_size: Input image size
            pretrained: Use pretrained weights
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.pretrained = pretrained

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = self._create_model()

        # Training state
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model_state = None

        logger.info(f"Initialized trainer with {model_name} on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _create_model(self) -> nn.Module:
        """Create and configure the model"""
        model = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            # img_size=self.img_size
        )
        return model.to(self.device)

    def train_epoch(
            self,
            train_loader: DataLoader,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Step scheduler if it's not ReduceLROnPlateau
            if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(
            self,
            val_loader: DataLoader,
            criterion: nn.Module
    ) -> Tuple[float, float, Dict]:
        """Validate for one epoch"""
        self.model.eval()

        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for data, targets in pbar:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # Update progress bar
                avg_loss = running_loss / (len(all_predictions) // targets.size(0))
                pbar.set_postfix({'Val Loss': f'{avg_loss:.4f}'})

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * accuracy_score(all_targets, all_predictions)

        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )

        metrics = {
            'accuracy': epoch_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'targets': all_targets
        }

        return epoch_loss, epoch_acc, metrics

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            optimizer_name: str = 'adamw',
            scheduler_name: str = 'cosine',
            save_dir: str = './checkpoints',
            save_best: bool = True,
            early_stopping_patience: int = 10,
            gradient_clip: Optional[float] = None
    ) -> Dict:
        """
        Complete training pipeline

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay
            optimizer_name: Optimizer ('adam', 'adamw', 'sgd')
            scheduler_name: Scheduler ('cosine', 'step', 'plateau', 'none')
            save_dir: Directory to save checkpoints
            save_best: Save best model
            early_stopping_patience: Early stopping patience
            gradient_clip: Gradient clipping value
        """

        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        optimizer = self._create_optimizer(optimizer_name, lr, weight_decay)

        # Setup scheduler
        scheduler = self._create_scheduler(scheduler_name, optimizer, epochs)

        # Setup criterion
        criterion = nn.CrossEntropyLoss()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }

        best_val_acc = 0.0
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Optimizer: {optimizer_name}, Scheduler: {scheduler_name}")
        logger.info(f"Learning rate: {lr}, Weight decay: {weight_decay}")

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, scheduler
            )

            # Validate
            val_loss, val_acc, val_metrics = self.validate_epoch(val_loader, criterion)

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            # Step scheduler for ReduceLROnPlateau
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

            # Apply gradient clipping if specified
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

            # Log epoch results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Val Precision: {val_metrics['precision']:.4f}, "
                        f"Val Recall: {val_metrics['recall']:.4f}, "
                        f"Val F1: {val_metrics['f1']:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                if save_best:
                    self.best_model_state = self.model.state_dict().copy()
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'model_name': self.model_name,
                        'num_classes': self.num_classes,
                        'img_size': self.img_size
                    }
                    torch.save(checkpoint, save_path / 'best_model.pth')
                    logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history
                }
                torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch + 1}.pth')

        # Save final history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Load best model
        if save_best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")

        return history

    def _create_optimizer(self, optimizer_name: str, lr: float, weight_decay: float):
        """Create optimizer"""
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay,
                             momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self, scheduler_name: str, optimizer, epochs: int):
        """Create learning rate scheduler"""
        if scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
        elif scheduler_name.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        elif scheduler_name.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        # Learning Rate
        ax3.plot(epochs, history['lr'], 'g-')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)

        # Best metrics
        best_train_acc = max(history['train_acc'])
        best_val_acc = max(history['val_acc'])
        ax4.bar(['Best Train Acc', 'Best Val Acc'], [best_train_acc, best_val_acc])
        ax4.set_title('Best Accuracies')
        ax4.set_ylabel('Accuracy (%)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate(self, test_loader: DataLoader, plot_confusion_matrix: bool = True):
        """Evaluate model on test set"""
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc='Evaluating'):
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )

        logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1: {f1:.4f}")

        # Plot confusion matrix
        if plot_confusion_matrix:
            cm = confusion_matrix(all_targets, all_predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probs
        }

    def save_model(self, filepath: str):
        """Save model state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'img_size': self.img_size
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")


# Example usage and utility functions
def create_sample_dataset(size=1000, num_classes=2, img_size=224):
    """Create a sample dataset for demonstration"""

    class SampleDataset(Dataset):
        def __init__(self, size=size, num_classes=num_classes, img_size=img_size):
            self.size = size
            self.num_classes = num_classes
            self.img_size = img_size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Generate random image and label
            image = torch.randn(3, self.img_size, self.img_size)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return image, label

    return SampleDataset()


if __name__ == "__main__":
    # Example usage

    # Create sample datasets
    train_dataset = create_sample_dataset(size=1000, num_classes=2, img_size=512)
    val_dataset = create_sample_dataset(size=200, num_classes=2, img_size=512)
    test_dataset = create_sample_dataset(size=200, num_classes=2, img_size=512)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize trainer
    trainer = TimmModelTrainer(
        model_name='convnext_tiny',
        num_classes=2,
        img_size=512,
        pretrained=True
    )

    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,  # Small number for demo
        lr=1e-3,
        optimizer_name='adamw',
        scheduler_name='cosine',
        save_dir='./checkpoints',
        early_stopping_patience=5
    )

    # Plot training history
    trainer.plot_training_history(history, save_path='training_history.png')

    # Evaluate on test set
    test_results = trainer.evaluate(test_loader)

    # Save final model
    trainer.save_model('final_model.pth')

    print("Training completed!")