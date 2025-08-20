#!/usr/bin/env python

import os
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
import argparse
import warnings
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)
from pytorch_lightning.loggers import WandbLogger

from models_custom.self_supervised import SelfSupervisedMAE
from augmentations.augmentation_composer import (
    get_pretrain_augmentations,
    get_val_augmentations,
)
from data.datamodule import PretrainDataModule
from data.pretrain_split import get_pretrain_split_config
from yucca.pipeline.configuration.configure_paths import detect_version
from utils.utils import setup_seed, SimplePathConfig
import logging


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Train MONAI MAE with PyTorch Lightning")

    # Required arguments
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to output directory for models and logs",
    )
    parser.add_argument(
        "--pretrain_data_dir",
        type=str,
        required=True,
        help="Path to pretraining data directory",
    )

    # Model parameters
    parser.add_argument("--img_size", type=int, default=96, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Vision transformer patch size")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Ratio of patches to mask")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--input_channels", type=int, default=1, help="Number of input channels")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")

    # Hardware configuration
    parser.add_argument("--num_devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--precision", type=str, default="32", help="Training precision")

    # Augmentation
    parser.add_argument(
        "--augmentation_preset",
        type=str,
        choices=["all", "basic", "none"],
        default="none",
        help="Augmentation preset to use"
    )

    # Training control
    parser.add_argument("--fast_dev_run", action="store_true", help="Fast development run")
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=10, #25
        help="Save a checkpoint every N epochs",
    )
    parser.add_argument("--new_version", action="store_true", help="Create new version directory")

    # Experiment naming
    parser.add_argument(
        "--experiment",
        type=str,
        default="mae_experiment",
        help="Name of experiment for logging"
    )

    args = parser.parse_args()

    print(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    print("ARGS:", args)

    # Set up directory structure
    train_data_dir = args.pretrain_data_dir
    model_name = "mae_monai"

    # Path where logs, checkpoints etc is stored
    save_dir = os.path.join(
        args.save_dir, "models", os.path.basename(train_data_dir), model_name
    )
    versions_dir = os.path.join(save_dir, "versions")
    continue_from_most_recent = not args.new_version
    version = detect_version(versions_dir, continue_from_most_recent)
    version_dir = os.path.join(versions_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

    # Configure training environment
    seed = setup_seed(continue_from_most_recent)

    # Set up data splits
    path_config = SimplePathConfig(train_data_dir=train_data_dir)
    splits_config = get_pretrain_split_config(
        method="simple_train_val_split",
        idx=0,
        split_ratio=0.01,  # We use 1% of data for validation split
        path_config=path_config,
    )

    # Configuration dictionary
    config = {
        # Experiment information
        "experiment": args.experiment,
        "model_name": model_name,
        "version": version,
        # Directories
        "save_dir": save_dir,
        "train_data_dir": train_data_dir,
        "version_dir": version_dir,
        # Reproducibility
        "seed": seed,
        # Model parameters
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "mask_ratio": args.mask_ratio,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "input_channels": args.input_channels,
        # Training parameters
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "effective_batch_size": args.accumulate_grad_batches * args.num_devices * args.batch_size,
        "precision": args.precision,
        "augmentation_preset": args.augmentation_preset,
        # Hardware configuration
        "num_devices": args.num_devices,
        "num_workers": args.num_workers,
        # Dataset metrics
        "train_dataset_size": len(splits_config.train(0)),
        "val_dataset_size": len(splits_config.val(0)),
        # Trainer specific params
        "fast_dev_run": args.fast_dev_run,
        "limit_val_batches": args.limit_val_batches,
        "limit_train_batches": args.limit_train_batches,
        "overfit_batches": args.overfit_batches,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "accumulate_grad_batches": args.accumulate_grad_batches,
    }

    print(
        f"Starting training for {config['epochs']} epochs "
        f"with {config['train_dataset_size']} training datapoints, {config['val_dataset_size']} validation datapoints, "
        f"and an effective batch size of {config['effective_batch_size']}"
    )

    # Set up data augmentation
    train_transforms = get_pretrain_augmentations(
        [args.img_size, args.img_size, args.img_size], args.augmentation_preset
    )
    val_transforms = get_val_augmentations()

    # Create data module (using the same configuration as in pretrain.py)
    data = PretrainDataModule(
        patch_size=[args.img_size, args.img_size, args.img_size],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        splits_config=splits_config,
        split_idx=0,
        train_data_dir=train_data_dir,
        composed_train_transforms=train_transforms,
        composed_val_transforms=val_transforms,
    )

    # Create the MONAI MAE model
    model = SelfSupervisedMAE(
        input_channels=args.input_channels,
        img_size=args.img_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
    )

    print("-" * 50, "SELECTED CONFIGURATIONS", "-" * 50)
    print(f"Model: SelfSupervisedMAE (MONAI)")
    print(f"Image size: {args.img_size}")
    print(f"Patch size: {args.patch_size}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Depth: {args.depth}")
    print(f"Num heads: {args.num_heads}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Train_data_dir: {train_data_dir}")
    print("-" * 130)

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=version_dir,
        filename="{epoch:02d}",
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_last=True,
        save_top_k=5,
        monitor="val_loss"
    )

    # Initialize wandb logging
    wandb.init(
        project="fomo25",
        name=f"{config['experiment']}_version_{config['version']}",
        config=config,
    )

    # Create wandb logger for Lightning
    wandb_logger = WandbLogger(
        project="fomo25",
        name=f"{config['experiment']}_version_{config['version']}",
        entity="matejgazda-technical-university-of-kosice",
        log_model=True,
    )

    # Create trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        strategy="ddp" if config["num_devices"] > 1 else "auto",
        devices=config["num_devices"],
        num_nodes=1,
        default_root_dir=config["save_dir"],
        max_epochs=config["epochs"],
        precision=config["precision"],
        fast_dev_run=config["fast_dev_run"],
        limit_val_batches=config["limit_val_batches"],
        limit_train_batches=config["limit_train_batches"],
        overfit_batches=config["overfit_batches"],
        check_val_every_n_epoch=config["check_val_every_n_epoch"],
        num_sanity_val_steps=0 if config["overfit_batches"] > 0 else 2,
        accumulate_grad_batches=config["accumulate_grad_batches"],
    )

    logging.info('-----------MAE PRETRAINING STARTED-----------')
    logging.info(f'Model: {model.__class__.__name__}')
    logging.info(f'Datamodule: {data}')

    # Start training
    trainer.fit(model=model, datamodule=data)

    if torch.cuda.is_available():
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Close the wandb logging session
    wandb.finish()

    print(f"Training completed! Model saved in: {version_dir}")
    print(f"To extract encoder later, use: model.get_encoder()")


if __name__ == "__main__":
    main()
