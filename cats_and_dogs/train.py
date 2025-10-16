import time

import numpy as np
import torch
from data import init_dataloader, init_dataset
from models import SimpleClassifier


TRAIN_DATASET_PATH = "../data/train_11k"
VAL_DATASET_PATH = "../data/val"
MODEL_NAME = "simple_model"

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 3


def train_model(
        model, 
        train_loader, 
        val_loader, 
        loss_fn, 
        opt, 
        device, 
        n_epochs,
        save_model_name
):
    """Training the model

    Args:
        model: model to train
        train_loader: dataloader for train data
        val_loader (torch.nn.utils.Dataloader): dataloader for validation data
        loss_fn: loss function to be used
        opt: optimizer to be used
        device: device used for training
        n_epochs: number of epochs to train
    """
    train_loss = []
    val_loss = []
    val_accuracy = []
    top_val_accuracy = -1
    best_model = None
    model = model.to(device)

    print("Start training...")
    for epoch in range(n_epochs):
        ep_train_loss = []
        ep_val_loss = []
        ep_val_accuracy = []
        start_time = time.time()

        model.train(True)
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            opt.zero_grad()
            predicts = model(X_batch)
            loss = loss_fn(predicts, y_batch)

            loss.backward()
            opt.step()
            ep_train_loss.append(loss.item())

        model.train(False)
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                preds = model(X_batch)
                validation_loss = loss_fn(preds, y_batch)

                ep_val_loss.append(validation_loss.item())
                preds_np = np.argmax(preds.detach().cpu().numpy(), 1).ravel()
                gt = y_batch.detach().cpu().numpy().ravel()
                hits = np.array(preds_np == gt)
                ep_val_accuracy.append(hits.astype(np.float32).mean())

        print(f"Epoch {epoch + 1} of {n_epochs} took {time.time() - start_time:.3f}s")

        train_loss.append(np.mean(ep_train_loss))
        val_loss.append(np.mean(ep_val_loss))
        val_accuracy.append(np.mean(ep_val_accuracy))

        print(f"\t  training loss: {train_loss[-1]:.6f}")
        print(f"\tvalidation loss: {val_loss[-1]:.6f}")
        print(f"\tvalidation accuracy: {100 * val_accuracy[-1]:.1f}")
        if val_accuracy[-1] > top_val_accuracy:
            best_model = model
            top_val_accuracy = val_accuracy[-1]

    torch.save(
        best_model.state_dict(),
        f"../models/{save_model_name}_{np.round(top_val_accuracy, 2)}.pt",
    )

    return train_loss, val_loss, val_accuracy, best_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataset = init_dataset(TRAIN_DATASET_PATH)
    train_loader = init_dataloader(train_dataset, BATCH_SIZE)

    val_dataset = init_dataset(VAL_DATASET_PATH)
    val_loader = init_dataloader(val_dataset, BATCH_SIZE)

    train_loss, val_loss, val_accuracy, best_model = train_model(
        model, train_loader, val_loader, loss_fn, opt, device, NUM_EPOCHS, MODEL_NAME
    )


if __name__ == "__main__":
    main()