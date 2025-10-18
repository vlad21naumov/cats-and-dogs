import numpy as np
import torch
from data import init_dataloader, init_dataset
from models import SimpleClassifier

TEST_DATASET_PATH = "../data/test_labeled"
MODEL_PATH = "../models/simple_model_0.5600000023841858.pt"

BATCH_SIZE = 128


@torch.no_grad()
def infer_model(model, test_loader, device, subset="test"):
    """Inference of the model

    Args:
        model: model to infer
        test_loader: dataloader for test data
        device: device used for inference
        subset: just for prettier printing. Defaults to "test".
    """
    model.train(False)
    test_batch_acc = []

    print("Start testing...")
    for X_batch, y_batch in test_loader:
        logits = model(X_batch.to(device))
        y_pred = logits.max(1)[1].data
        test_batch_acc.append(np.mean((y_batch.cpu() == y_pred.cpu()).numpy()))

    test_accuracy = np.mean(test_batch_acc)

    print("Results:")
    print(f"    {subset} accuracy: {test_accuracy * 100:.2f} %")


def main():
    model = SimpleClassifier()

    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(checkpoint)

    test_dataset = init_dataset(TEST_DATASET_PATH)
    test_loader = init_dataloader(test_dataset, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_model(model, test_loader, device)


if __name__ == "__main__":
    main()
