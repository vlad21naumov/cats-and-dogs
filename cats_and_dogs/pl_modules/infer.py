import numpy as np
import pytorch_lightning as pl

from pl_modules.data import MyDataModule
from pl_modules.model import ImageClassifier


def main():
    dm = MyDataModule()

    model = ImageClassifier.load_from_checkpoint(
        "../models/epoch=02-val_loss=0.6046.ckpt"
    )
    trainer = pl.Trainer(accelerator="gpu", devices="auto")

    accs = trainer.predict(model, datamodule=dm)
    print(f"Test accuracy: {np.mean(accs):.2f}")


if __name__ == "__main__":
    main()
