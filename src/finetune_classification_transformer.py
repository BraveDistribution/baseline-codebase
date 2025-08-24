from augmentations.finetune_augmentation_presets import get_finetune_augmentation_params
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from pathlib import Path
from fire import Fire

from utils.utils import SimplePathConfig
from yucca.modules.data.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)
from yucca.pipeline.configuration.split_data import get_split_config
from mato_models.models import ClassificationFineTuner, RegressionFineTuner, SegmentationFineTuner, ClassificationFinetuner2, RegressionFinetuner2
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.callbacks.loggers import YuccaLogger
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset
from data.dataset import FOMODataset


def train(
    data_dir: Path | str,
    save_checkpoint_dir: Path | str,
    model_checkpoint: Path | str,
    num_epochs: int = 100,
    batch_size: int = 8,
    patch_size: int = 96,
    split_method: str = "simple_train_val_split",
    split_param: float = 0.2,
    split_idx: int = 0,
    num_workers: int = 12,
    experiment_name: str = "Classification Finetuning",
    task_type: str = "classification",
):
    print("--- Training Parameters ---")
    for key, value in locals().items():
        print(f"{key:<20}: {value}")
    print("--------------------------")
    if split_method == "kfold":
        split_param = int(split_param)
    elif split_method == "simple_train_val_split":
        split_param = float(split_param)
    else:
        split_param = split_param

    if task_type == "classification":
        num_modalities = 4
    elif task_type == "regression":
        num_modalities = 2
    elif task_type == 'segmentation':
        num_modalities = 3
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_checkpoint_dir,
        filename="best-checkpoint",
        monitor="val/loss",
        mode="min",
        save_top_k=5,
    )
    # aug_params = get_finetune_augmentation_params("all")
    aug_params = get_finetune_augmentation_params("basic")
    task_type_preset = "classification" if task_type == "regression" else task_type
    augmenter = YuccaAugmentationComposer(
        patch_size=[patch_size, patch_size, patch_size],
        task_type_preset=task_type_preset,
        parameter_dict=aug_params,
        deep_supervision=False,
    )

    path_config = SimplePathConfig(train_data_dir=data_dir)
    splits_config = get_split_config(
        method=split_method,
        param=split_param,
        path_config=path_config,
    )

    data_module = YuccaDataModule(
        train_dataset_class=(
            YuccaTrainDataset if task_type == "segmentation" else FOMODataset
        ),
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        patch_size=[patch_size, patch_size, patch_size],
        batch_size=batch_size,
        train_data_dir=data_dir,
        image_extension=".npy",
        task_type=task_type,
        splits_config=splits_config,
        split_idx=split_idx,
        num_workers=num_workers,
        val_sampler=None,
    )

    wandb_logger = WandbLogger(
        project="fomo-finetuning",
        name=experiment_name,
        log_model=True,
    )

    if task_type == "classification":
        # model = ClassificationFineTuner.load_from_checkpoint(
        #     str(model_checkpoint),
        #     num_classes=1,
        #     in_channels=num_modalities,
        #     multichannel_strategy="copy",
        #     freeze_encoder=True,
        #     learning_rate=1e-4,
        #     max_epochs=50,
        # )
        model = ClassificationFinetuner2.load_from_pretrained(
            checkpoint_path=str(model_checkpoint),
            num_classes=1,
            in_channels=num_modalities,
            freeze_encoder=False,
            learning_rate=1e-4, # We discussed using a lower LR for fine-tuning
            max_epochs=50
        )
    elif task_type == "regression":
        model = RegressionFinetuner2.load_from_checkpoint(
            str(model_checkpoint),
            in_channels=num_modalities,
            freeze_encoder=True,
            learning_rate=1e-4,
            max_epochs=50,
            strict=False
        )
    elif task_type == 'segmentation':
        model = SegmentationFineTuner.load_from_checkpoint(
            str(model_checkpoint),
            num_classes=1,
            in_channels=3,
            freeze_encoder=True,
            learning_rate=1e-4,
            max_epochs=50,
        )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback],
        logger=[wandb_logger],
        accelerator="gpu",
        precision='32-true',
        limit_train_batches=30,
        accumulate_grad_batches=5,
        log_every_n_steps=15,
        
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(train)
