#!/usr/bin/env python
"""
Hierarchical Finetuning Script

This script implements finetuning for hierarchical multi-resolution models:
- RegressionHierarchicalFinetuner: For brain age regression tasks
- ClassificationHierarchicalFinetuner: For classification tasks (future)
- SegmentationHierarchicalFinetuner: For segmentation tasks (future)

The script uses HierarchicalDataset to provide both local (high-res) and
global (low-res) views of the same data for multi-scale feature learning.

Usage Examples:
    # Brain age regression with hierarchical model (local checkpoint required)
    python finetune_hierarchical.py \
        --task_id 3 \
        --model_type regression \
        --local_checkpoint /path/to/contrastive_pretrained.ckpt \
        --global_checkpoint /path/to/global_encoder_pretrained.ckpt \
        --epochs 100 \
        --batch_size 4

    # Classification with hierarchical model (future, local checkpoint required)
    python finetune_hierarchical.py \
        --task_id 1 \
        --model_type classification \
        --local_checkpoint /path/to/contrastive_pretrained.ckpt \
        --epochs 50

Author: AI Assistant
Date: August 2025
"""

import os
import argparse
import logging
import torch
import pytorch_lightning as pl
from typing import Dict, Any

# CRITICAL: Set up deterministic algorithms with warn_only BEFORE any other imports
# This prevents the max_pool3d deterministic error from MONAI SwinUNETR
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("🔧 Early deterministic setup: warn_only=True enabled")
except Exception as e:
    print(f"🔧 Early deterministic setup failed: {e}")
    # Fallback to disabling deterministic algorithms entirely
    try:
        torch.use_deterministic_algorithms(False)
        print("🔧 Fallback: Deterministic algorithms disabled")
    except:
        pass

# Lightning components
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Data and augmentation
from yucca.modules.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.callbacks.loggers import YuccaLogger
from yucca.pipeline.configuration.split_data import get_split_config

# Project imports
from data.dataset import HierarchicalDataset
from data.task_configs import task1_config, task2_config, task3_config
from kamil_models.models import RegressionHierarchicalFinetuner
from augmentations.finetune_augmentation_presets import get_finetune_augmentation_params
from utils.utils import SimplePathConfig, setup_seed, find_checkpoint
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p as ensure_dir_exists


class HierarchicalConfig:
    """
    Configuration class for hierarchical finetuning.
    Organizes all parameters in a clear, readable structure.
    """

    def __init__(self, args):
        # === TASK CONFIGURATION ===
        self.task_id = args.task_id
        self.task_config = self._get_task_config(args.task_id)
        self.task_type = self.task_config["task_type"]
        self.task_name = self.task_config["task_name"]
        self.num_classes = self.task_config["num_classes"]
        self.num_modalities = len(self.task_config["modalities"])
        self.labels = self.task_config["labels"]

        # === MODEL CONFIGURATION ===
        self.model_type = args.model_type
        self.global_encoder = args.global_encoder
        self.patch_size = (args.patch_size,) * 3
        self.feature_size = args.feature_size
        self.freeze_global_encoder = args.freeze_global_encoder

        # LoRA configuration for local encoder (defaults from classification transformer)
        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha

        # Model-specific parameters for regression tasks
        if self.task_type == "regression":
            self.target_mean = args.target_mean
            self.target_std = args.target_std
            self.predict_uncertainty = args.predict_uncertainty
            self.mixup_alpha = args.mixup_alpha
            self.mixup_prob = args.mixup_prob

        # === TRAINING CONFIGURATION ===
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.dropout_rate = args.dropout_rate
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.train_batches_per_epoch = args.train_batches_per_epoch

        # === DATA CONFIGURATION ===
        self.local_data_dir = args.local_data_dir
        self.global_data_dir = args.global_data_dir
        self.augmentation_preset = args.augmentation_preset

        # === AGE BALANCING CONFIGURATION ===
        # Only relevant for regression tasks with age prediction
        self.use_balanced_dataset = getattr(args, 'use_balanced_dataset', False)
        self.n_age_bins = getattr(args, 'n_age_bins', 8)
        self.balancing_strategy = getattr(args, 'balancing_strategy', 'oversample')
        age_range_list = getattr(args, 'age_range', [20.0, 100.0])
        self.age_range = tuple(age_range_list)  # Convert list to tuple
        self.oversample_factor = getattr(args, 'oversample_factor', 1.0)

        # === EXPERIMENT CONFIGURATION ===
        self.save_dir = args.save_dir
        self.experiment_name = args.experiment_name
        self.local_checkpoint = args.local_checkpoint
        self.global_checkpoint = args.global_checkpoint
        self.continue_training = args.continue_training
        self.precision = args.precision

        # === HARDWARE CONFIGURATION ===
        self.num_devices = args.num_devices
        self.num_workers = args.num_workers
        self.accelerator = args.accelerator

        # === SPLIT CONFIGURATION ===
        self.split_method = args.split_method
        self.split_param = args.split_param
        self.split_idx = args.split_idx

        # === DERIVED CONFIGURATIONS ===
        self.effective_batch_size = self.num_devices * self.batch_size
        self.max_iterations = self.epochs * self.train_batches_per_epoch

        # Experiment naming
        self.full_experiment_name = f"{self.experiment_name}_{self.task_type}_Task00{self.task_id}"

        # Global encoder configuration for hierarchical model
        self.global_config = {
            "model_name": self.global_encoder,
            "num_modalities": self.num_modalities,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "task_type": self.task_type
        }

    def _get_task_config(self, task_id: int) -> Dict[str, Any]:
        """Get task configuration based on task ID."""
        if task_id == 1:
            return task1_config
        elif task_id == 2:
            return task2_config
        elif task_id == 3:
            return task3_config
        else:
            raise ValueError(f"Unknown task_id: {task_id}. Supported: 1, 2, 3")

    def print_summary(self):
        """Print a comprehensive summary of the configuration."""
        print("\n" + "="*80)
        print("HIERARCHICAL FINETUNING CONFIGURATION SUMMARY")
        print("="*80)

        print(f"\n📋 TASK INFORMATION:")
        print(f"   Task ID: {self.task_id}")
        print(f"   Task Name: {self.task_name}")
        print(f"   Task Type: {self.task_type}")
        print(f"   Num Classes: {self.num_classes}")
        print(f"   Num Modalities: {self.num_modalities}")

        print(f"\n🏗️  MODEL ARCHITECTURE:")
        print(f"   Model Type: {self.model_type}")
        print(f"   Global Encoder: {self.global_encoder}")
        print(f"   Patch Size: {self.patch_size}")
        print(f"   Feature Size: {self.feature_size}")
        print(f"   Freeze Global Encoder: {self.freeze_global_encoder}")
        print(f"   LoRA Rank: {self.lora_r}")
        print(f"   LoRA Alpha: {self.lora_alpha}")

        if hasattr(self, 'target_mean'):
            print(f"   Target Mean: {self.target_mean}")
            print(f"   Target Std: {self.target_std}")
            print(f"   MixUp Alpha: {self.mixup_alpha}")

        print(f"\n🎯 TRAINING PARAMETERS:")
        print(f"   Learning Rate: {self.learning_rate}")
        print(f"   Weight Decay: {self.weight_decay}")
        print(f"   Dropout Rate: {self.dropout_rate}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Effective Batch Size: {self.effective_batch_size}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batches/Epoch: {self.train_batches_per_epoch}")
        print(f"   Max Iterations: {self.max_iterations}")

        print(f"\n📁 DATA CONFIGURATION:")
        print(f"   Local Data Dir: {self.local_data_dir}")
        print(f"   Global Data Dir: {self.global_data_dir}")
        print(f"   Augmentation: {self.augmentation_preset}")
        print(f"   Split Method: {self.split_method}")
        print(f"   Split Param: {self.split_param}")

        print(f"\n⚖️  AGE BALANCING:")
        if hasattr(self, 'use_balanced_dataset') and self.use_balanced_dataset:
            print(f"   Balanced Dataset: ✅ ENABLED")
            print(f"   Age Bins: {self.n_age_bins}")
            print(f"   Strategy: {self.balancing_strategy}")
            print(f"   Age Range: {self.age_range}")
            print(f"   Oversample Factor: {self.oversample_factor}")
            print(f"   Note: Validation uses unweighted loss for unbiased evaluation")
        else:
            print(f"   Balanced Dataset: ❌ DISABLED (standard dataset)")
            print(f"   Note: Using original age distribution")

        print(f"\n🖥️  HARDWARE SETUP:")
        print(f"   Accelerator: {self.accelerator}")
        print(f"   Num Devices: {self.num_devices}")
        print(f"   Num Workers: {self.num_workers}")
        print(f"   Precision: {self.precision}")

        print(f"\n🚀 EXPERIMENT:")
        print(f"   Experiment Name: {self.full_experiment_name}")
        print(f"   Save Directory: {self.save_dir}")
        if self.local_checkpoint:
            print(f"   Local Checkpoint: {self.local_checkpoint}")
        if self.global_checkpoint:
            print(f"   Global Checkpoint: {self.global_checkpoint}")

        print("="*80)


def create_hierarchical_model(config: HierarchicalConfig) -> pl.LightningModule:
    """
    Create the appropriate hierarchical model based on configuration.
    Hierarchical models always require pretrained weights for both local and global encoders.

    Args:
        config: Hierarchical configuration object

    Returns:
        Configured PyTorch Lightning model
    """
    # Validate that required checkpoints are provided
    if not config.local_checkpoint or not os.path.exists(config.local_checkpoint):
        raise ValueError(f"Local checkpoint is required for hierarchical models. "
                        f"Provided: {config.local_checkpoint}")

    if config.model_type == "regression":
        print(f"Loading hierarchical regression model from local checkpoint: {config.local_checkpoint}")
        if config.global_checkpoint:
            print(f"Using global checkpoint: {config.global_checkpoint}")

        model = RegressionHierarchicalFinetuner.load_from_pretrained(
            local_checkpoint=config.local_checkpoint,
            in_channels=config.num_modalities,
            target_mean=config.target_mean,
            target_std=config.target_std,
            global_config=config.global_config,
            global_checkpoint=config.global_checkpoint,
            img_size=config.patch_size,
            feature_size=config.feature_size,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            learning_rate=config.learning_rate,
            dropout_rate=config.dropout_rate,
            max_epochs=config.epochs,
            predict_uncertainty=config.predict_uncertainty,
            weight_decay=config.weight_decay,
            mixup_alpha=config.mixup_alpha,
            mixup_prob=config.mixup_prob,
            freeze_global_encoder=config.freeze_global_encoder,
            # Age balancing parameters
            use_balanced_dataset=config.use_balanced_dataset,
            n_age_bins=config.n_age_bins,
            balancing_strategy=config.balancing_strategy,
            age_range=config.age_range,
            oversample_factor=config.oversample_factor,
        )

    elif config.model_type == "classification":
        # TODO: Implement ClassificationHierarchicalFinetuner
        raise NotImplementedError("ClassificationHierarchicalFinetuner not yet implemented")

    elif config.model_type == "segmentation":
        # TODO: Implement SegmentationHierarchicalFinetuner
        raise NotImplementedError("SegmentationHierarchicalFinetuner not yet implemented")

    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    return model


def create_data_module(config: HierarchicalConfig) -> YuccaDataModule:
    """
    Create data module with HierarchicalDataset for multi-resolution training.
    Uses HierarchicalAgeBalancedDataset for training if age balancing is enabled,
    but always uses regular HierarchicalDataset for validation to keep it unbiased.

    Args:
        config: Hierarchical configuration object

    Returns:
        Configured YuccaDataModule with appropriate datasets
    """
    # Set up augmentations
    aug_params = get_finetune_augmentation_params(config.augmentation_preset)

    # Use classification preset for regression (as in original code)
    task_type_preset = "classification" if config.task_type == "regression" else config.task_type

    augmenter = YuccaAugmentationComposer(
        patch_size=config.patch_size,
        task_type_preset=task_type_preset,
        parameter_dict=aug_params,
        deep_supervision=False,
    )

    # Set up data splits
    train_data_dir = os.path.join(config.local_data_dir, config.task_name)
    path_config = SimplePathConfig(train_data_dir=train_data_dir)

    if config.split_method == "kfold":
        split_param = int(config.split_param)
    else:
        split_param = float(config.split_param)

    splits_config = get_split_config(
        method=config.split_method,
        param=split_param,
        path_config=path_config,
    )

    # Import the balanced dataset
    from data.dataset import HierarchicalDataset, HierarchicalAgeBalancedDataset
    from functools import partial

    # Choose dataset class based on configuration
    if hasattr(config, 'use_balanced_dataset') and config.use_balanced_dataset and config.task_type == "regression":
        print(f"🔄 Using HierarchicalAgeBalancedDataset with {config.n_age_bins} age bins")
        print("📊 Note: Balancing applies only during training, validation remains unbiased")

        # Create dataset class with age balancing parameters
        DatasetClass = partial(
            HierarchicalAgeBalancedDataset,
            local_data_dir=config.local_data_dir,
            global_data_dir=config.global_data_dir,
            n_bins=getattr(config, 'n_age_bins', 8),
            balancing_strategy=getattr(config, 'balancing_strategy', 'oversample'),
            age_range=getattr(config, 'age_range', (20.0, 100.0)),
            oversample_factor=getattr(config, 'oversample_factor', 1.0),
            verbose=True
        )
    else:
        print("📊 Using standard HierarchicalDataset")
        DatasetClass = partial(
            HierarchicalDataset,
            local_data_dir=config.local_data_dir,
            global_data_dir=config.global_data_dir,
        )

    # Create data module - use single dataset class for both train and validation
    # doesn't support separate validation dataset classes
    data_module = YuccaDataModule(
        train_dataset_class=DatasetClass,
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        train_data_dir=train_data_dir,
        image_extension=".npy",
        task_type=config.task_type,
        splits_config=splits_config,
        split_idx=config.split_idx,
        num_workers=config.num_workers,
        val_sampler=None,
    )

    return data_module


def create_callbacks(config: HierarchicalConfig, version_dir: str) -> list:
    """
    Create training callbacks for monitoring and checkpointing.

    Args:
        config: Hierarchical configuration object
        version_dir: Directory for saving checkpoints

    Returns:
        List of configured callbacks
    """
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=version_dir,
        filename="hierarchical_best_{epoch:02d}_{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        enable_version_counter=False,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping for regression tasks
    if config.task_type == "regression":
        early_stopping = EarlyStopping(
            monitor="val/correlation",
            mode="max",
            patience=25,
            verbose=True,
            strict=False,  # Allow missing metrics during initial epochs
        )
        callbacks.append(early_stopping)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    return callbacks


def create_loggers(config: HierarchicalConfig, version: int) -> list:
    """
    Create loggers for experiment tracking.

    Args:
        config: Hierarchical configuration object
        version: Experiment version number

    Returns:
        List of configured loggers
    """
    loggers = []

    # # Yucca logger for local logging
    # yucca_logger = YuccaLogger(
    #     save_dir=config.save_dir,
    #     version=version,
    #     steps_per_epoch=config.train_batches_per_epoch,
    # )
    # loggers.append(yucca_logger)

    # Wandb logger for cloud logging and visualization
    wandb_logger = WandbLogger(
        project="hierarchical-finetuning",
        name=f"{config.full_experiment_name}_v{version}",
        log_model=True,
        tags=[
            f"task_{config.task_id}",
            config.task_type,
            config.model_type,
            f"global_{config.global_encoder}",
            "hierarchical"
        ]
    )
    loggers.append(wandb_logger)

    return loggers


def setup_experiment_directory(config: HierarchicalConfig) -> tuple[str, int]:
    """
    Set up experiment directory structure and determine version.

    Args:
        config: Hierarchical configuration object

    Returns:
        Tuple of (version_directory, version_number)
    """
    from yucca.pipeline.configuration.configure_paths import detect_version

    # Create base save directory
    save_dir = os.path.join(config.save_dir, config.task_name, "hierarchical", config.global_encoder)
    ensure_dir_exists(save_dir)

    # Handle versioning
    continue_from_most_recent = config.continue_training
    version = detect_version(save_dir, continue_from_most_recent)
    version_dir = os.path.join(save_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

    return version_dir, version


def apply_deterministic_settings():
    """Apply deterministic settings that handle non-deterministic CUDA operations."""
    try:
        # Ensure our warn_only setting is still active
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("✓ Deterministic settings confirmed: warn_only=True for unsupported operations")
        return True

    except Exception as e:
        print(f"⚠️  Failed to apply deterministic settings: {e}")
        # Last resort: disable deterministic algorithms completely
        try:
            torch.use_deterministic_algorithms(False)
            print("✓ Fallback: Deterministic algorithms disabled to prevent errors")
            return False
        except:
            return False


def setup_deterministic_training():
    """
    Set up deterministic training configuration that handles CUDA operations
    without deterministic implementations.
    """
    return apply_deterministic_settings()
def train_hierarchical_model(config: HierarchicalConfig):
    """
    Main training function for hierarchical models.

    Args:
        config: Hierarchical configuration object
    """
    # Print configuration summary
    config.print_summary()

    # Set up experiment directory
    version_dir, version = setup_experiment_directory(config)
    print(f"\n📁 Experiment directory: {version_dir}")

    # Set up reproducibility and deterministic training
    seed = setup_seed(config.continue_training)
    print(f"🌱 Using seed: {seed}")

    # Set up deterministic algorithms with proper handling for CUDA operations
    deterministic_enabled = setup_deterministic_training()

    # Look for existing checkpoint if continuing training
    ckpt_path = find_checkpoint(version_dir, config.continue_training) if config.continue_training else None
    if ckpt_path:
        print(f"🔄 Resuming from checkpoint: {ckpt_path}")

    # Create model
    print(f"\n🏗️  Creating hierarchical {config.model_type} model...")
    model = create_hierarchical_model(config)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")

    # Create data module
    print(f"\n📊 Setting up hierarchical data module...")
    data_module = create_data_module(config)

    # Calculate dataset sizes
    train_samples = len(data_module.splits_config.train(config.split_idx))
    val_samples = len(data_module.splits_config.val(config.split_idx))
    print(f"   Training samples: {train_samples}")
    print(f"   Validation samples: {val_samples}")
    print(f"   Local data: {config.local_data_dir}")
    print(f"   Global data: {config.global_data_dir}")

    # Create callbacks and loggers
    callbacks = create_callbacks(config, version_dir)
    loggers = create_loggers(config, version)

    # Create trainer
    print(f"\n🚀 Setting up trainer...")

    # Apply deterministic settings right before trainer creation
    print(f"   Applying deterministic settings before trainer initialization...")
    apply_deterministic_settings()

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        devices=config.num_devices,
        precision=config.precision,
        callbacks=callbacks,
        logger=loggers,
        limit_train_batches=config.train_batches_per_epoch,
        val_check_interval=min(1.0, 100 / config.train_batches_per_epoch),  # Validate every 100 steps or once per epoch
        log_every_n_steps=max(1, config.train_batches_per_epoch // 10),  # Log 10 times per epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=deterministic_enabled,
        enable_checkpointing=True,
    )

    # Log configuration to wandb
    if wandb_logger := next((l for l in loggers if isinstance(l, WandbLogger)), None):
        wandb_logger.experiment.config.update({
            "hierarchical_config": {
                "task_id": config.task_id,
                "task_type": config.task_type,
                "model_type": config.model_type,
                "global_encoder": config.global_encoder,
                "local_data_dir": config.local_data_dir,
                "global_data_dir": config.global_data_dir,
                "patch_size": config.patch_size,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "total_params": total_params,
                "trainable_params": trainable_params,
            }
        })

    # Start training
    print(f"\n🎯 Starting hierarchical training...")
    print(f"   Model: {config.model_type} with {config.global_encoder}")
    print(f"   Task: {config.task_name} (ID: {config.task_id})")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size} x {config.num_devices} = {config.effective_batch_size}")

    try:
        # Ensure deterministic settings are still active before training
        print(f"\n🔧 Final confirmation of deterministic settings before training...")
        apply_deterministic_settings()

        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {version_dir}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hierarchical Finetuning Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === TASK CONFIGURATION ===
    task_group = parser.add_argument_group("Task Configuration")
    task_group.add_argument(
        "--task_id", type=int, required=True, choices=[1, 2, 3],
        help="Task ID (1: FOMO1 classification, 2: FOMO2 classification, 3: FOMO3 regression)"
    )
    task_group.add_argument(
        "--model_type", type=str, default="regression", choices=["regression", "classification", "segmentation"],
        help="Type of hierarchical model to train"
    )

    # === MODEL CONFIGURATION ===
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--global_encoder", type=str, default="unet_b",
        choices=["unet_b", "unet_b_lw_dec", "unet_xl", "unet_xl_lw_dec",
                "mednext_s3", "mednext_m3", "mednext_l3"],
        help="Global encoder architecture"
    )
    model_group.add_argument(
        "--patch_size", type=int, default=96,
        help="Patch size for both local and global views"
    )
    model_group.add_argument(
        "--feature_size", type=int, default=24,
        help="Feature size for SwinUNETR local encoder"
    )
    model_group.add_argument(
        "--freeze_global_encoder", action="store_true",
        help="Freeze global encoder weights during training"
    )

    # LoRA configuration
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument("--lora_r", type=int, default=128, help="LoRA rank")
    lora_group.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")

    # Regression-specific parameters
    regression_group = parser.add_argument_group("Regression Configuration")
    regression_group.add_argument("--target_mean", type=float, default=61.87, help="Target mean for normalization")
    regression_group.add_argument("--target_std", type=float, default=15.089118845634706, help="Target std for normalization")
    regression_group.add_argument("--predict_uncertainty", action="store_true", help="Predict uncertainty")
    regression_group.add_argument("--mixup_alpha", type=float, default=0.4, help="MixUp alpha parameter")
    regression_group.add_argument("--mixup_prob", type=float, default=0.5, help="MixUp probability")

    # === TRAINING CONFIGURATION ===
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    training_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    training_group.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    training_group.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    training_group.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    training_group.add_argument("--train_batches_per_epoch", type=int, default=100, help="Batches per epoch")

    # === DATA CONFIGURATION ===
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--local_data_dir", type=str,
        default="/home/mg873uh/Projects_kb/data/finetuning_preproc/",
        help="Local (high-resolution) data directory"
    )
    data_group.add_argument(
        "--global_data_dir", type=str,
        default="/home/mg873uh/Projects_kb/data/finetuning_preproc/Unified_2.6667mm_float16",
        help="Global (low-resolution) data directory"
    )
    data_group.add_argument(
        "--augmentation_preset", type=str, default="basic", choices=["all", "basic", "none"],
        help="Augmentation preset"
    )

    # === AGE BALANCING CONFIGURATION ===
    balance_group = parser.add_argument_group("Age Balancing Configuration")
    balance_group.add_argument(
        "--use_balanced_dataset", action="store_true",
        help="Use HierarchicalAgeBalancedDataset for training (regression only)"
    )
    balance_group.add_argument(
        "--n_age_bins", type=int, default=8,
        help="Number of age bins for balancing (default: 8 for range 20-100)"
    )
    balance_group.add_argument(
        "--balancing_strategy", type=str, default="oversample",
        choices=["oversample", "undersample", "hybrid"],
        help="Strategy for balancing age distribution"
    )
    balance_group.add_argument(
        "--age_range", type=float, nargs=2, default=[20.0, 100.0],
        help="Expected age range for binning (min max)"
    )
    balance_group.add_argument(
        "--oversample_factor", type=float, default=1.0,
        help="Factor to multiply target samples per bin (1.0 = equal bin sizes)"
    )

    # === SPLIT CONFIGURATION ===
    split_group = parser.add_argument_group("Data Split Configuration")
    split_group.add_argument("--split_method", type=str, default="simple_train_val_split", help="Split method")
    split_group.add_argument("--split_param", type=str, default="0.2", help="Split parameter")
    split_group.add_argument("--split_idx", type=int, default=0, help="Split index for k-fold")

    # === EXPERIMENT CONFIGURATION ===
    experiment_group = parser.add_argument_group("Experiment Configuration")
    experiment_group.add_argument(
        "--save_dir", type=str, default="./data/models",
        help="Directory to save models and results"
    )
    experiment_group.add_argument(
        "--experiment_name", type=str, default="hierarchical_experiment",
        help="Experiment name for logging"
    )
    experiment_group.add_argument(
        "--local_checkpoint", type=str, required=True,
        help="Path to pretrained ContrastiveTransformer checkpoint for local encoder (REQUIRED)"
    )
    experiment_group.add_argument(
        "--global_checkpoint", type=str, default=None,
        help="Path to pretrained checkpoint for global encoder"
    )
    experiment_group.add_argument(
        "--continue_training", action="store_true",
        help="Continue from most recent checkpoint"
    )
    experiment_group.add_argument(
        "--precision", type=str, default="bf16-mixed", choices=["32", "16", "bf16-mixed"],
        help="Training precision"
    )

    # === HARDWARE CONFIGURATION ===
    hardware_group = parser.add_argument_group("Hardware Configuration")
    hardware_group.add_argument("--num_devices", type=int, default=1, help="Number of devices")
    hardware_group.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    hardware_group.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"], help="Accelerator type")

    return parser.parse_args()


def main():
    """Main entry point for hierarchical finetuning."""
    # Set up logging
    logging.getLogger().setLevel(logging.INFO)

    # Parse arguments and create configuration
    args = parse_arguments()
    config = HierarchicalConfig(args)

    # Validate configuration
    assert config.patch_size[0] % 8 == 0, f"Patch size must be divisible by 8, got {config.patch_size[0]}"

    # Start training
    train_hierarchical_model(config)


if __name__ == "__main__":
    main()