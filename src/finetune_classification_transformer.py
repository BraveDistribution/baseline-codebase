from augmentations.finetune_augmentation_presets import get_finetune_augmentation_params
from data import datamodule
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from pathlib import Path
from fire import Fire

from utils.utils import SimplePathConfig
from yucca.modules.data.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)
from yucca.pipeline.configuration.split_data import get_split_config
from mato_models.models import ClassificationFineTuner, RegressionFineTuner, SegmentationFineTuner, ClassificationFinetuner2, RegressionFinetuner2, RegressionFinetuner3,  SegmentationProtoNet, RegressionFinetuner4, ClassificationFinetunerMTL
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.callbacks.loggers import YuccaLogger
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset
from data.dataset import FOMODataset, FOMODatasetWithSeg


from collections import Counter
import torch



import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from collections import Counter
import numpy as np
from typing import Optional, Literal
import matplotlib.pyplot as plt



import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np
import logging
from typing import Optional, Literal, Union
from os.path import join
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler

class AgeBalancedWeights:
    """Keep only the weighting logic."""
    def __init__(self, dataset, num_bins=15, strategy='uniform', temperature=0.8):
        self.dataset = dataset
        self.num_bins = num_bins
        self.strategy = strategy
        self.temperature = temperature
        self.labels = self._get_labels()
        self.weights = self._create_weights()

    def _get_labels(self):
        if hasattr(self.dataset, 'labels'):
            return self.dataset.labels
        labels = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            if isinstance(sample, dict) and 'label' in sample:
                v = sample['label']
                labels.append(v.item() if torch.is_tensor(v) else v)
        return labels

    def _create_weights(self):
        labels_tensor = torch.tensor(self.labels, dtype=torch.float32)
        q = torch.linspace(0, 1, self.num_bins + 1)
        labels_jitter = labels_tensor + 1e-4 * torch.randn_like(labels_tensor)
        bins = torch.quantile(labels_jitter, q)
        bins, _ = torch.sort(torch.unique(bins))
        num_bins = max(1, bins.numel() - 1)
        binned = torch.bucketize(labels_tensor, bins, right=True) - 1
        binned = binned.clamp(min=0, max=num_bins - 1)
        counts = torch.bincount(binned, minlength=num_bins).float()
        eps = 1.0
        if self.strategy == 'uniform':
            alpha = 1.0
        elif self.strategy == 'sqrt':
            alpha = 0.5
        elif self.strategy == 'smooth':
            alpha = max(0.1, min(1.0, self.temperature))
        else:
            alpha = 0.5
        raw_w = (counts + eps).pow(-alpha)
        w = raw_w / raw_w.mean()
        w = torch.minimum(w, w.median() * 3.0)
        w = 0.3 * torch.ones_like(w) + 0.7 * w   # blend
        return w[binned].double()

class YuccaDataModuleWithBalancing(pl.LightningDataModule):
    """
    Extended YuccaDataModule with age-balanced sampling for regression tasks.

    This module extends the standard YuccaDataModule to include age-balanced sampling
    when task_type is 'regression' and age balancing is enabled.
    """

    def __init__(
        self,
        batch_size: int,
        patch_size: tuple,
        allow_missing_modalities: Optional[bool] = False,
        image_extension: Optional[str] = None,
        composed_train_transforms: Optional[torchvision.transforms.Compose] = None,
        composed_val_transforms: Optional[torchvision.transforms.Compose] = None,
        num_workers: Optional[int] = None,
        overwrite_predictions: bool = False,
        pred_data_dir: Optional[str] = None,
        pred_include_cases: Optional[list] = None,
        pred_save_dir: Optional[str] = None,
        pre_aug_patch_size: Optional[Union[list, tuple]] = None,
        p_oversample_foreground: Optional[float] = 0.33,
        splits_config: Optional[object] = None,  # SplitConfig
        split_idx: Optional[int] = None,
        task_type: Optional[str] = None,
        test_dataset_class: Optional[torch.utils.data.Dataset] = None,  # YuccaTestDataset
        train_data_dir: Optional[str] = None,
        train_dataset_class: Optional[torch.utils.data.Dataset] = None,  # YuccaTrainDataset or FOMODataset
        train_sampler: Optional[object] = None,  # Can be InfiniteRandomSampler or our AgeBalancedSampler
        val_sampler: Optional[object] = None,
        # New parameters for age balancing
        use_age_balancing: bool = False,
        age_balance_config: Optional[dict] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.image_extension = image_extension
        self.task_type = task_type

        self.split_idx = split_idx
        self.splits_config = splits_config
        self.train_data_dir = train_data_dir

        self.allow_missing_modalities = allow_missing_modalities
        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms
        self.pre_aug_patch_size = pre_aug_patch_size
        self.p_oversample_foreground = p_oversample_foreground

        # Prediction settings
        self.pred_include_cases = pred_include_cases
        self.overwrite_predictions = overwrite_predictions
        self.pred_data_dir = pred_data_dir
        self.pred_save_dir = pred_save_dir

        # Age balancing settings
        self.use_age_balancing = use_age_balancing
        self.age_balance_config = age_balance_config or {
            'num_bins': 15,
            'strategy': 'uniform',
            'temperature': 0.8,
            'visualize': False
        }

        # Set default values
        self.num_workers = max(0, int(torch.get_num_threads() - 1)) if num_workers is None else num_workers
        self.val_num_workers = self.num_workers
        self.test_dataset_class = test_dataset_class
        self.train_sampler = train_sampler
        self.train_dataset_class = train_dataset_class
        self.val_sampler = val_sampler

        logging.info(f"Using {self.num_workers} workers")
        logging.info(f"Using dataset class: {self.train_dataset_class} for train/val")
        if self.use_age_balancing and self.task_type == 'regression':
            logging.info(f"Age balancing enabled with config: {self.age_balance_config}")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        logging.info(f"Setting up data for stage: {stage}")

        if stage == "fit":
            assert self.train_data_dir is not None
            assert self.split_idx is not None
            assert self.splits_config is not None
            assert self.task_type is not None

            self.train_samples = [join(self.train_data_dir, i) for i in self.splits_config.train(self.split_idx)]
            self.val_samples = [join(self.train_data_dir, i) for i in self.splits_config.val(self.split_idx)]

            if len(self.train_samples) < 100:
                logging.info(f"Training on samples: {self.train_samples}")

            if len(self.val_samples) < 100:
                logging.info(f"Validating on samples: {self.val_samples}")

            # Create training dataset
            self.train_dataset = self.train_dataset_class(
                self.train_samples,
                composed_transforms=self.composed_train_transforms,
                patch_size=self.pre_aug_patch_size if self.pre_aug_patch_size is not None else self.patch_size,
                task_type=self.task_type,
                allow_missing_modalities=self.allow_missing_modalities,
                p_oversample_foreground=self.p_oversample_foreground,
            )

            # Create validation dataset
            self.val_dataset = self.train_dataset_class(
                self.val_samples,
                composed_transforms=self.composed_val_transforms,
                patch_size=self.patch_size,
                task_type=self.task_type,
                allow_missing_modalities=self.allow_missing_modalities,
                p_oversample_foreground=self.p_oversample_foreground,
            )

            # Visualize age distribution if requested
            if (self.use_age_balancing and
                self.task_type == 'regression' and
                self.age_balance_config.get('visualize', False)):
                self._visualize_age_distribution()

        if stage == "predict":
            assert self.pred_data_dir is not None, "`pred_data_dir` is required in inference"
            assert self.pred_save_dir is not None, "`pred_save_dir` is required in inference"
            assert self.image_extension is not None, "`image_extension` is required in inference"

            self.pred_dataset = self.test_dataset_class(
                self.pred_data_dir,
                pred_save_dir=self.pred_save_dir,
                overwrite_predictions=self.overwrite_predictions,
                suffix=self.image_extension,
                pred_include_cases=self.pred_include_cases,
            )

    def train_dataloader(self):
        logging.info(f"Starting training with data from: {self.train_data_dir}")

        # Determine which sampler to use
        if self.use_age_balancing and self.task_type == 'regression':
            # Use age-balanced sampler for regression tasks
            sampler = AgeBalancedSampler(
                self.train_dataset,
                num_bins=self.age_balance_config.get('num_bins', 15),
                strategy=self.age_balance_config.get('strategy', 'sqrt'),
                temperature=self.age_balance_config.get('temperature', 0.8),
                infinite=True  # Match InfiniteRandomSampler behavior
            )
            logging.info("Using age-balanced sampler for training")
        elif self.train_sampler is not None:
            # Use provided sampler (e.g., InfiniteRandomSampler)
            sampler = self.train_sampler(self.train_dataset)
        else:
            sampler = None

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
            shuffle=sampler is None,
        )

    def val_dataloader(self):
        # Validation typically doesn't need balancing
        sampler = self.val_sampler(self.val_dataset) if self.val_sampler is not None else None
        return DataLoader(
            self.val_dataset,
            num_workers=self.val_num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
        )

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        logging.info("Starting inference")
        from functools import partial
        # Assuming single_case_collate is defined elsewhere
        single_case_collate = lambda x: x[0]  # Simple implementation
        return DataLoader(
            self.pred_dataset,
            num_workers=self.num_workers,
            batch_size=1,
            collate_fn=single_case_collate
        )

    def _visualize_age_distribution(self):
        """Visualize the age distribution in training data."""
        try:
            import matplotlib.pyplot as plt

            if hasattr(self.train_dataset, 'labels'):
                labels = self.train_dataset.labels

                plt.figure(figsize=(10, 4))
                plt.hist(labels, bins=self.age_balance_config.get('num_bins', 15),
                        alpha=0.7, color='blue', edgecolor='black')
                plt.title('Training Set Age Distribution')
                plt.xlabel('Age (years)')
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)

                # Add statistics
                plt.axvline(np.mean(labels), color='red', linestyle='--',
                           label=f'Mean: {np.mean(labels):.1f}')
                plt.axvline(np.median(labels), color='green', linestyle='--',
                           label=f'Median: {np.median(labels):.1f}')
                plt.legend()

                plt.tight_layout()
                plt.show()

                logging.info(f"Age statistics - Min: {min(labels):.1f}, Max: {max(labels):.1f}, "
                           f"Mean: {np.mean(labels):.1f}, Median: {np.median(labels):.1f}")
        except ImportError:
            logging.warning("Matplotlib not available for visualization")


# Usage example
def create_balanced_datamodule(config):
    """
    Example function to create a balanced data module for brain age regression.
    """
    from yucca.data.data_module import YuccaDataModule  # Import original if needed

    datamodule = YuccaDataModuleWithBalancing(
        batch_size=config['batch_size'],
        patch_size=config['patch_size'],
        train_data_dir=config['train_data_dir'],
        splits_config=config['splits_config'],
        split_idx=config['split_idx'],
        task_type='regression',  # For brain age regression
        train_dataset_class=FOMODataset,  # Your dataset class
        composed_train_transforms=config['train_transforms'],
        composed_val_transforms=config['val_transforms'],
        num_workers=config.get('num_workers', 4),
        # Enable age balancing
        use_age_balancing=True,
        age_balance_config={
            'num_bins': 16,  # For 20-100 age range, creates ~5-year bins
            'strategy': 'uniform',  # Strong rebalancing to fix 40-70 clustering
            'temperature': 0.8,  # Fine-tune based on results
            'visualize': True  # Show distribution on first run
        }
    )

    return datamodule


def add_age_balancing_to_existing_datamodule(datamodule, age_balance_config=None):
    orig = datamodule.train_dataloader

    def balanced_train_dataloader(self):
        if getattr(self, "task_type", None) == "regression":
            cfg = age_balance_config or {'num_bins': 15, 'strategy': 'uniform', 'temperature': 0.8}
            weights = AgeBalancedWeights(
                self.train_dataset, cfg['num_bins'], cfg['strategy'], cfg['temperature']
            ).weights
            sampler = WeightedRandomSampler(weights, num_samples=len(self.train_dataset), replacement=True)
            return DataLoader(
                self.train_dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=torch.cuda.is_available(),
                sampler=sampler,
                shuffle=False,
            )
        return orig()

    datamodule.train_dataloader = balanced_train_dataloader.__get__(datamodule, type(datamodule))
    return datamodule



def train(
    data_dir: Path | str,
    save_checkpoint_dir: Path | str,
    model_checkpoint: Path | str,
    num_epochs: int = 100,
    batch_size: int = 8,
    patch_size: int = 96,
    split_method: str = "simple_train_val_split",
    split_param: float = 0.2,  # Use all data for training
    split_idx: int = 0,
    num_workers: int = 6,
    experiment_name: str = "Classification Finetuning",
    task_type: str = "classification",
    n_splits: int = 0,
    aug_setup: str = "basic",
):
    print("--- Training Parameters ---")
    for key, value in locals().items():
        print(f"{key:<20}: {value}")
    print("--------------------------")
    if split_method == "kfold":
        split_param = n_splits
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

    monitor = "val/loss"
    if split_param < 0.05:
        monitor = "val/loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_checkpoint_dir,
        filename="best-checkpoint-{task_type}-{val/loss:.4f}",
        monitor=monitor,
        mode="min",
        save_top_k=5,
        save_last=True,
    )

    if task_type == "regression":
        early_stopping = EarlyStopping(
            monitor="val/mae",
            mode="min",
            patience=25,
            verbose=True,
            strict=False,  # Allow missing metrics during initial epochs
        )

    # aug_params = get_finetune_augmentation_params("all")
    if aug_setup not in ['basic','all']:
        raise AttributeError("Invalid augmentation setup")
    aug_params = get_finetune_augmentation_params(aug_setup)
    # aug_params["crop"] = True
    # aug_params["random_crop"] = False
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

    if task_type == "segmentation":
        dataset = YuccaTrainDataset
    elif task_type == 'regression':
        dataset = FOMODataset
    elif task_type == 'classification':
        dataset = FOMODatasetWithSeg

    data_module = YuccaDataModule(
        train_dataset_class=(
            dataset
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

    if task_type == 'regression':
        data_module = add_age_balancing_to_existing_datamodule(
            data_module,  # ← Use the variable you created
            age_balance_config={
                'num_bins': 10,
                'strategy': 'smooth',
                'temperature': 0.85
            }
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
        # model = ClassificationFinetunerMTL.load_from_pretrained(
        #     checkpoint_path=str(model_checkpoint),
        #     img_size=(96, 96, 96),
        #     num_classes=2,
        #     in_channels=4,
        #     feature_size=24,
        #     backbone_lr=2e-5,
        #     head_lr=2e-4,
        #     cls_loss_weight=1.0,
        #     seg_loss_weight=0.5
        # )
    elif task_type == "regression":
        # model = RegressionFinetuner2.load_from_checkpoint(
        #     str(model_checkpoint),
        #     in_channels=num_modalities,
        #     freeze_encoder=True,
        #     learning_rate=1e-4,
        #     max_epochs=50,
        #     strict=False
        # )
        model = RegressionFinetuner4.load_from_pretrained(
            checkpoint_path=str(model_checkpoint),
            in_channels=2,
            target_min=18.0,
            target_max=120.0,
            target_mean=4.105172539442826,
            target_std=0.28682802940837737,
            feature_size=24,
            original_target_mean=61.87,
            # target_std=15.089118845634706,
        )

    elif task_type == 'segmentation':
        model = SegmentationFineTuner.load_from_pretrained(
            str(model_checkpoint),
            num_classes=2,
            in_channels=3,
            feature_size=24,
            freeze_encoder=True,
            learning_rate=1e-4,
            max_epochs=50,
            out_channels=2,
        )

    callbacks=[checkpoint_callback]
    if task_type == "regression":
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=[wandb_logger],
        accelerator="gpu",
        precision='16-mixed',
        limit_train_batches=30,
        accumulate_grad_batches=5,
        log_every_n_steps=15,
        # check_val_every_n_epoch=100,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,  # Skip validation sanity check
        check_val_every_n_epoch=None,  # Disable validation entirely
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(train)
