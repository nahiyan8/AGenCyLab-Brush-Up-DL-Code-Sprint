# # PyTorch Lightning CIFAR10 ~94% Baseline Tutorial
# 
# * **Author:** PL team
# * **License:** CC BY-SA
# * **Generated:** 2022-04-28T08:05:29.967173
# 
# Train a Resnet to 94% accuracy on Cifar10!

import pandas as pd
import seaborn as sn
from IPython.core.display import display
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from data import cifar10_dm
from configs import cfg
from models import LitResnet, SWAResnet

def main():
    seed_everything(cfg.RANDOM_SEED)

    # ### Resnet

    # ### Lightning Module

    model = LitResnet(lr=0.05)

    trainer = Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=cfg.NUM_DEVICES,  # limiting got iPython runs
        logger=CSVLogger(save_dir=cfg.PATH_LOGS),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")

    # ### Bonus: Use [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407) to get a boost on performance

    swa_model = SWAResnet(model.model, lr=0.01)
    swa_model.datamodule = cifar10_dm

    swa_trainer = Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=cfg.NUM_DEVICES,  # limiting got iPython runs
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        logger=CSVLogger(save_dir=cfg.PATH_LOGS),
    )

    swa_trainer.fit(swa_model, cifar10_dm)
    swa_trainer.test(swa_model, datamodule=cifar10_dm)

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")

if __name__ == '__main__':
    main()
