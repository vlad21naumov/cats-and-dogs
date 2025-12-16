import os

import hydra
import mlflow.pyfunc
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from PIL import Image


class ConvClassifier(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(6),
            torch.nn.Flatten(),
            torch.nn.Linear(4608, 96),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(96, num_classes, bias=False),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class InferenceModule(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = ConvClassifier(num_classes)

    def forward(self, batch):
        pred = self.model(batch)
        return pred.argmax(dim=1)


class CatsDogsClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.device = torch.device("cpu")
        self.model = InferenceModule(num_classes=2)

        ckpt = torch.load(context.artifacts["model_weights"], map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_key = k[len("model.") :]
            else:
                new_key = k
            new_state_dict[new_key] = v

        self.model.model.load_state_dict(new_state_dict)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, context, model_input):
        images = []
        for img_data in model_input:
            img_path = str(img_data["data"])

            if not os.path.exists(img_path):
                raise ValueError(f"File {img_path} not found")

            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0)
            images.append(img_tensor)

        batch = torch.cat(images, dim=0)
        with torch.no_grad():
            preds = self.model(batch)

        return preds.numpy()


@hydra.main(version_base=None, config_path=".", config_name="conf")
def main(config: DictConfig):
    mlflow.pyfunc.save_model(
        path="cats_dogs_model",
        python_model=CatsDogsClassifier(),
        artifacts={"model_weights": config["weights_path"]},
    )


if __name__ == "__main__":
    main()
