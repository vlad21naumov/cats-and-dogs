import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from cats_and_dogs.pl_modules.classifiers import ConvClassifier
from cats_and_dogs.pl_modules.data import MyDataModule
from cats_and_dogs.pl_modules.model import ImageClassifier


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: DictConfig):
    pl.seed_everything(42)
    dm = MyDataModule(config)
    model = ImageClassifier(
        ConvClassifier(num_classes=config["model"]["num_classes"]),
        lr=config["training"]["lr"],
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
            project=config["logging"]["project"],
            name=config["logging"]["name"],
            save_dir=config["logging"]["save_dir"],
        ),
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
            save_top_k=5,
            every_n_train_steps=None,
            every_n_epochs=1,
        )
    )

    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     precision=32,
    #     max_epochs=10,
    #     accumulate_grad_batches=1,
    #     val_check_interval=1.0,
    #     overfit_batches=0,
    #     num_sanity_val_steps=4,
    #     deterministic=False,
    #     benchmark=False,
    #     gradient_clip_val=2.0,
    #     profiler=None,
    #     log_every_n_steps=1,
    #     detect_anomaly=False,
    #     enable_checkpointing=True,
    #     logger=loggers,
    #     callbacks=callbacks,
    # )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,  # to resolve warnings
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
