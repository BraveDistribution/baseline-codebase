#!/usr/bin/env python

import os
import torch
import pytorch_lightning as pl
import argparse
import warnings
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)
from pytorch_lightning.loggers import WandbLogger

from models.self_supervised import SelfSupervisedModel
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
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--model_name", type=str, default="unet_b_lw_dec")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument(
        "--mask_patch_size",
        type=int,
        default=4,
        help="i.e. MAE patch size, the masking unit.",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3, #Khamyl[Added]
        default=[240, 256, 256],
        help="Unified shapes of volumes." #Khamyl[Modified]: "The patch size of the 3D patches extracted from the whole volume.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None)
    parser.add_argument("--new_version", action="store_true")
    parser.add_argument("--optimizer", type=str, default="AdamW")

    parser.add_argument(
        "--augmentation_preset",
        type=str,
        choices=["all", "basic", "none"],
        default="none",
    )
    parser.add_argument("--loss_masked_tokens_only", default=False, action="store_true")

    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=3)
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None)
    parser.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=25,
        help="Save a checkpoint every N epochs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='default',
    )

    parser.add_argument(
        "--experiment", type=str, default="base_experiment", help="name of experiment"
    )

    args = parser.parse_args()

    assert all(x % 8 == 0 for x in args.patch_size), args.patch_size #Khamyl[Modif]: args.patch_size % 8 == 0, args.patch_size
    assert all(args.mask_patch_size < x for x in args.patch_size), ( #Khamyl[Modif]: assert args.mask_patch_size < args.patch_size
        f"mask_patch_size ({args.mask_patch_size}) must be less than all patch_size dimensions {args.patch_size}"
    )

    print(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    print("ARGS:", args)

    # Set up directory structure
    train_data_dir = args.pretrain_data_dir

    # Path where logs, checkpoints etc is stored
    save_dir = os.path.join(
        args.save_dir, "models", os.path.basename(train_data_dir), args.model_name
    )
    versions_dir = os.path.join(save_dir, "versions")
    continue_from_most_recent = not args.new_version
    version = detect_version(versions_dir, continue_from_most_recent)
    version_dir = os.path.join(versions_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

    # Configure training environment
    seed = setup_seed(continue_from_most_recent)

    # Create dataset splits
    path_config = SimplePathConfig(train_data_dir=train_data_dir)
    splits_config = get_pretrain_split_config(
        method="simple_train_val_split",
        idx=0,
        split_ratio=0.01,  # We use 1% of data for validation split
        path_config=path_config,
    )

    # configuration dictionary
    config = {
        # Experiment information
        "experiment": args.experiment,
        "model_name": args.model_name,
        "model_dimensions": "3D",
        "task_type": "self-supervised",
        "version": version,
        # Directories
        "save_dir": save_dir,
        "train_data_dir": train_data_dir,
        "version_dir": version_dir,
        # Reproducibility
        "seed": seed,
        # Model parameters
        "patch_size": args.patch_size, # Khamyl[Modified]: (args.patch_size,) * 3,
        "mask_patch_size": args.mask_patch_size,
        "mask_ratio": args.mask_ratio,
        "input_channels": 1,
        "num_classes": 1,
        "should_compile": args.compile,
        "compile_mode": args.compile_mode,
        "rec_loss_masked_only": args.loss_masked_tokens_only,
        # Training parameters
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "effective_batch_size": args.accumulate_grad_batches
        * args.num_devices
        * args.batch_size,
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
        "gradient_clip_val": 1.0
    }

    # Calculate training metrics based on the config
    steps_per_epoch = (
        int(config["train_dataset_size"] / config["effective_batch_size"])
        if config["overfit_batches"] == 0
        else config["overfit_batches"]
    )
    max_iterations = int(config["epochs"] * steps_per_epoch)
    config["steps_per_epoch"] = steps_per_epoch
    config["max_iterations"] = max_iterations

    print(
        f"Starting training with {max_iterations} max iterations over {config['epochs']} epochs "
        f"with {config['train_dataset_size']} training datapoints, {config['val_dataset_size']} validation datapoints, "
        f"and an effective batch size of {config['effective_batch_size']}"
    )

    # Set up data augmentation and datamodule
    train_transforms = get_pretrain_augmentations(
        config["patch_size"], args.augmentation_preset
    )
    val_transforms = get_val_augmentations()

    model = None
    data = None

    print("-" * 50, "SELECTED CONFIGURATIONS", "-" * 50)
    print(f"Model type: {args.model}")
    print(f"Patch size: {config['patch_size']}")
    print(f"Train_data_dir: {train_data_dir}")
    print("-" * 130)

    if args.model == 'contrastive':
        print('CREATING CONTRASTIVE DATA MODULE')
        data = PretrainDataModule(
            patch_size=config["patch_size"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            splits_config=splits_config,
            split_idx=0,
            train_data_dir=train_data_dir,
            composed_train_transforms=None,
            composed_val_transforms=None,
            dataset='contrastive',
            crop=False,
        )
    else:
        data = PretrainDataModule(
            patch_size=config["patch_size"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            splits_config=splits_config,
            split_idx=0,
            train_data_dir=train_data_dir,
            composed_train_transforms=train_transforms,
            composed_val_transforms=val_transforms,
        )

    if args.model == 'default':
        model = SelfSupervisedModel(
            model_name=config["model_name"],
            config=config,
            epochs=config["epochs"],
            warmup_epochs=config["warmup_epochs"],
            learning_rate=config["learning_rate"],
            optimizer=config["optimizer"],
            steps_per_epoch=config["steps_per_epoch"],
            num_classes=config["num_classes"],
            input_channels=config["input_channels"],
            patch_size=config["patch_size"],
            mask_patch_size=config["mask_patch_size"],
            mask_ratio=config["mask_ratio"],
            should_compile=config["should_compile"],
            compile_mode=config["compile_mode"],
            rec_loss_masked_only=config["rec_loss_masked_only"],
        )

    if args.model == 'contrastive':
        from models_custom.models import SwinUNETRPretraining
        model = SwinUNETRPretraining(
            # Model architecture
            img_size=config.get("img_size", (96, 96, 96)),  # Should match your patch_size
            in_channels=config["input_channels"],
            out_channels=config.get("out_channels", config["input_channels"]),  # Same as input for MAE
            feature_size=config.get("feature_size", 24),
            spatial_dims=config.get("spatial_dims", 3),
            depths=config.get("depths", (2, 2, 2, 2)),
            num_heads=config.get("num_heads", (3, 6, 12, 24)),
            window_size=config.get("window_size", (7, 7, 7)),

            # Regularization
            drop_rate=config.get("drop_rate", 0.0),
            attn_drop_rate=config.get("attn_drop_rate", 0.0),
            dropout_path_rate=config.get("dropout_path_rate", 0.0),
            use_checkpoint=config.get("use_checkpoint", True),

            # MAE specific
            mask_ratio=config["mask_ratio"],
            mask_patch_size=config["mask_patch_size"],

            # Contrastive specific
            projection_dim=config.get("projection_dim", 128),
            projection_hidden_dim=config.get("projection_hidden_dim", 2048),
            temperature=config.get("temperature", 0.07),

            # Training parameters
            learning_rate=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01),
            warmup_epochs=config["warmup_epochs"],
            epochs=config["epochs"],
            steps_per_epoch=config["steps_per_epoch"],
            cosine_period_ratio=config.get("cosine_period_ratio", 1.0),

            # Loss weights
            mae_weight=config.get("mae_weight", 1.0),
            contrastive_weight=config.get("contrastive_weight", 5.0),

            # Other
            normalize=config.get("normalize", True),
            use_v2=config.get("use_v2", False),
            disable_image_logging=config.get("disable_image_logging", False),
            debug_losses=config.get("debug_losses", False),
        )

    # Initialize wandb logging
    wandb.init(
        project="fomo25",
        name=f"{config['experiment']}_version_{config['version']}",
    )

    # Create wandb logger for Lightning
    wandb_logger = WandbLogger(
        project="fomo25",
        name=f"{config['experiment']}_version_{config['version']}",
        entity="matejgazda-technical-university-of-kosice",
        log_model=True,
    )

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=version_dir,
        filename="{epoch:02d}",
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_last=True,
        save_top_k=5,
        monitor="val/loss"
    )
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        strategy="ddp" if config["num_devices"] > 1 else "auto",
        num_nodes=1,
        devices=config["num_devices"],
        default_root_dir=config["save_dir"],
        max_epochs=config["epochs"],
        precision='16-mixed',
        fast_dev_run=config["fast_dev_run"],
        limit_val_batches=config["limit_val_batches"],
        limit_train_batches=config["limit_train_batches"],
        overfit_batches=config["overfit_batches"],
        check_val_every_n_epoch=config["check_val_every_n_epoch"],
        num_sanity_val_steps=0 if config["overfit_batches"] > 0 else 2,
        accumulate_grad_batches=config["accumulate_grad_batches"],
    )

    logging.info('-----------MATO-PRETRAINING STARTED-----------')
    logging.info(f'datamodule={data}')
    logging.info(f"args.model: '{args.model}'")

    trainer.fit(model=model, datamodule=data)
    trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Close the wandb logging session
    wandb.finish()


if __name__ == "__main__":
    main()
