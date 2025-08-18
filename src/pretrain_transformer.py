import os
from pathlib import Path
import pytorch_lightning as pl

from typing import Sequence
from fire import Fire

from mato_data.data import ContrastiveDataModule
from mato_models.models import ContrastiveTransformer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def train(
    data_dir: str | Path, 
    mae_mask_ratio: float = 0.6,
    model_checkpoint_dir: str | Path = 'checkpoints',
    epochs: int = 100,
    mask_patch_size: int = 4,
    patch_size: Sequence[int] = (96, 96, 96),
    batch_size: int = 10,
    learning_rate: float = 1e-4,
    loss_masked_tokens_only: bool = True,
    accumulate_grad_batches: int = 3,
    checkpoint_every_n_epoch: int = 1,
    experiment_name: str = "default_experiment",
    resume_from_checkpoint: str | Path | None = None,
) -> None:
    save_dir: str | Path = Path(model_checkpoint_dir) / experiment_name
    transforms = None
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          filename="{epoch:02d}-{step}", every_n_train_steps=50)
    data_module = ContrastiveDataModule(data_dir=data_dir, transforms=transforms, batch_size=batch_size)
    if resume_from_checkpoint:
        model = ContrastiveTransformer.load_from_checkpoint(resume_from_checkpoint)
    else:
        model = ContrastiveTransformer(patch_size, learning_rate)
    wandb_logger = WandbLogger(
        project="PretrainingFOMO25",
        name=experiment_name,
        entity="matejgazda-technical-university-of-kosice",
        log_model=True,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        strategy="ddp",
        precision="bf16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        gradient_clip_val=1,
        gradient_clip_algorithm='norm',
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    Fire(train)