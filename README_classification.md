# SwinUNETR Classification Model

This document describes the `SwinUNETRClassification` model, which is a classification variant of the SwinUNETR pretraining model. It uses the pretrained encoder and adds a classification head on top.

## Overview

The `SwinUNETRClassification` model consists of:
1. **Encoder**: SwinUNETR encoder (can be loaded from pretrained weights)
2. **Classification Head**: A neural network that maps encoder features to class predictions

## Key Features

- **Pretrained Weight Loading**: Load encoder weights from pretraining models
- **Flexible Training**: Option to freeze encoder or fine-tune end-to-end
- **Adaptive Learning Rates**: Different learning rates for encoder vs classifier
- **Standard Classification Metrics**: Automatic logging of loss and accuracy

## Usage

### 1. Basic Classification Model

```python
from src.models_custom.models import SwinUNETRClassification

# Create model from scratch
model = SwinUNETRClassification(
    num_classes=5,           # Number of classes in your dataset
    img_size=(96, 96, 96),   # Input image dimensions
    in_channels=1,           # Number of input channels
    feature_size=24,         # Feature dimensions
    learning_rate=1e-4,
    weight_decay=0.01,
)
```

### 2. Load Pretrained Weights (Frozen Encoder)

```python
# Only train the classification head
model = SwinUNETRClassification(
    num_classes=3,
    pretrained_weights_path="/path/to/pretrained/model.ckpt",
    freeze_encoder=True,     # Freeze encoder weights
    learning_rate=1e-3,      # Higher LR since only training classifier
)
```

### 3. Fine-tuning with Pretrained Weights

```python
# Train both encoder and classifier with different learning rates
model = SwinUNETRClassification(
    num_classes=10,
    pretrained_weights_path="/path/to/pretrained/model.ckpt",
    freeze_encoder=False,    # Train both encoder and classifier
    learning_rate=1e-4,      # Encoder gets 10x lower LR automatically
)
```

## Model Parameters

### Required Parameters
- `num_classes`: Number of output classes

### Architecture Parameters
- `img_size`: Input image dimensions (default: (96, 96, 96))
- `in_channels`: Number of input channels (default: 1)
- `feature_size`: Base feature dimension (default: 24)
- `spatial_dims`: Spatial dimensions (default: 3)
- `depths`: Transformer block depths (default: (2, 2, 2, 2))
- `num_heads`: Attention heads per layer (default: (2, 4, 8, 16))

### Classification-Specific Parameters
- `hidden_dim`: Hidden dimension in classifier (default: 512)
- `dropout_classifier`: Dropout rate in classifier (default: 0.3)

### Training Parameters
- `learning_rate`: Base learning rate (default: 1e-4)
- `weight_decay`: Weight decay for regularization (default: 0.01)
- `warmup_epochs`: Learning rate warmup epochs (default: 10)
- `epochs`: Total training epochs (default: 100)

### Pretrained Model Parameters
- `pretrained_weights_path`: Path to pretrained model checkpoint (default: None)
- `freeze_encoder`: Whether to freeze encoder weights (default: False)

## Training

### Data Format

The model expects batches in one of these formats:

**Dictionary format:**
```python
batch = {
    "image": torch.tensor,  # Shape: [B, C, H, W, D]
    "label": torch.tensor   # Shape: [B] with class indices
}
```

**Tuple format:**
```python
batch = (
    torch.tensor,  # Images: [B, C, H, W, D]
    torch.tensor   # Labels: [B] with class indices
)
```

### Training Loop with PyTorch Lightning

```python
import pytorch_lightning as pl

# Create model
model = SwinUNETRClassification(
    num_classes=5,
    learning_rate=1e-4,
)

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
)

# Train
trainer.fit(model, train_dataloader, val_dataloader)
```

## Model Methods

### Core Methods
- `forward(x)`: Full forward pass returning logits
- `forward_encoder(x)`: Forward through encoder only
- `load_pretrained_weights(path)`: Load pretrained encoder weights
- `freeze_encoder_weights()`: Freeze encoder parameters
- `unfreeze_encoder_weights()`: Unfreeze encoder parameters

### Training Steps
- `training_step()`: Training step with loss and accuracy computation
- `validation_step()`: Validation step with metrics logging
- `test_step()`: Test step for evaluation
- `predict_step()`: Prediction step returning probabilities and predictions

## Pretrained Weight Loading

The model can automatically load pretrained weights from the `SwinUNETRPretraining` model:

1. **Automatic Filtering**: Only encoder weights are loaded, classifier weights are ignored
2. **Key Mapping**: Handles different checkpoint formats and key naming conventions
3. **Graceful Failure**: If loading fails, training continues from scratch with a warning

### Supported Checkpoint Formats
- PyTorch Lightning checkpoints (`.ckpt` files)
- Regular PyTorch state dictionaries (`.pth` files)
- Checkpoints with or without `state_dict` wrapper

## Learning Rate Strategy

When `freeze_encoder=False`, the model uses different learning rates:
- **Encoder**: `learning_rate * 0.1` (10x lower for stability)
- **Classifier**: `learning_rate` (full learning rate for new weights)

## Example Training Script

See `example_classification.py` for complete examples including:
- Model creation with different configurations
- Training loop setup with PyTorch Lightning
- Pretrained weight loading examples
- Data loading patterns

## Integration with Existing Codebase

The classification model is designed to work seamlessly with your existing infrastructure:
- Compatible with your data preprocessing pipelines
- Uses same PyTorch Lightning framework as pretraining
- Can load weights from your existing pretraining checkpoints
- Follows same logging and monitoring patterns
