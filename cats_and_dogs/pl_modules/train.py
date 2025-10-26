import pytorch_lightning as pl
from classifiers import ConvClassifier
from model import ImageClassifier

from data import MyDataModule


def main():
    pl.seed_everything(42)
    dm = MyDataModule()
    model = ImageClassifier(
        ConvClassifier(num_classes=2),
        lr=1e-3,
    )

    loggers = [
        # pl.loggers.CSVLogger("./logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        # pl.loggers.MLFlowLogger(
        #     experiment_name="cats-and-dogs",
        #     run_name="conv-classifier",
        #     save_dir=".",
        #     tracking_uri="http://127.0.0.1:8080",
        # ),
        # pl.loggers.TensorBoardLogger(
        #     "./.logs/my-tb-logs", name=cfg.artifacts.experiment_name
        # ),
        pl.loggers.WandbLogger(
            project="cats-and-dogs",
            name="conv-model-exp-25fmsai",
            save_dir=".",
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath="../../models",
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=2,
        )
    )

    trainer = pl.Trainer(
        max_epochs=3,
        log_every_n_steps=1,  # to resolve warnings
        accelerator="cpu",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
