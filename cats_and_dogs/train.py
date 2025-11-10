import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pl_modules.classifiers import ConvClassifier
from pl_modules.data import MyDataModule
from pl_modules.model import ImageClassifier


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    pl.seed_everything(42)
    dm = MyDataModule(cfg=config)
    model = ImageClassifier(
        ConvClassifier(num_classes=config["model"]["num_classes"]),
        lr=config["training"]["lr"],
    )

    loggers = [
        # pl.loggers.CSVLogger("./logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        pl.loggers.MLFlowLogger(
            experiment_name=config["logging"]["experiment_name"],
            run_name=config["logging"]["run_name"],
            save_dir=config["logging"]["mlflow_save_dir"],
            tracking_uri=config["logging"]["tracking_uri"],
        ),
        # pl.loggers.TensorBoardLogger(
        #     "./.logs/my-tb-logs", name=cfg.artifacts.experiment_name
        # ),
        # pl.loggers.WandbLogger(
        #     project="cats-and-dogs",
        #     name="conv-model-exp-25fmsai",
        #     save_dir=".",
        # ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=config["model"]["model_local_path"],
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=config["model"]["save_top_k"],
        )
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=config["training"][
            "log_every_n_steps"
        ],  # to resolve warnings
        accelerator="cpu",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
