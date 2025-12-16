import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig

from pl_modules.data import MyDataModule
from pl_modules.model import ImageClassifier


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    dm = MyDataModule(config)

    model = ImageClassifier.load_from_checkpoint(config["inference"]["ckpt_path"])
    trainer = pl.Trainer(accelerator="cpu")

    accs = trainer.predict(model, datamodule=dm)
    print(f"Test accuracy: {np.mean(accs):.2f}")


if __name__ == "__main__":
    main()
