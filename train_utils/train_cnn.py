"""
    ======================================================================
                              Train CNN Script
    ======================================================================
    Name:          train_cnn.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    This script is responsible for training and evaluating the
    convolutional neural network for defect classification. It uses the
    folder structure defined under data/cnn/<dataset> and builds PyTorch
    datasets and dataloaders using the PCBCNNDataset class from
    dataset_cnn.py. It constructs the DefectCNN model from
    models/cnn_model.py, sets up the loss function (cross-entropy)
    and optimizer (Adam), and then runs through the training loop for
    a specified number of epochs. At each epoch it computes training
    loss, training accuracy, and validation accuracy and prints those
    values they can be monitored. It keeps track of the best
    validation accuracy and saves the corresponding model
    parameters to checkpoints_cnn/best_cnn.pth. When training is done,
    it reloads that best checkpoint and evaluates it on the test set,
    then returns a small dictionary with the best validation accuracy,
    the final test accuracy, and the path to the saved checkpoint.
------------------------------------------------Imports and Globals-----------------------------------------------------
"""
import os
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model_configuration.dataset_cnn import PCBCNNDataset
from models.cnn_model import DefectCNN
from utils.seed_utils import set_global_seed
import matplotlib.pyplot as plt
"""---------------------------------------------------Functions------------------------------------------------------"""
"""
    Function Name: train_and_evaluate_cnn
    Description:
        Trains a convolutional neural network (DefectCNN) for PCB defect
        classification and evaluates it on a held-out test set. This function
        sets the global random seed for reproducibility, builds train/val/test
        datasets using the PCBCNNDataset class, wraps them in DataLoaders,
        and applies appropriate image transforms for training and evaluation.
        It constructs the DefectCNN model, defines a cross-entropy loss
        function and an Adam optimizer, and runs the training loop for a
        specified number of epochs. At each epoch it reports training loss,
        training accuracy, and validation accuracy, and it tracks the model
        with the best validation accuracy. The best model checkpoint is saved
        to disk and later reloaded for final evaluation on the test set.
    Input:
        data_root (str, optional):
            Root directory of the CNN dataset organized in a folder-per-class
            structure, with split subfolders 'train', 'val', and 'test'.
            Default is "data/cnn".
        num_classes (int, optional):
            Number of defect classes in the dataset. This determines the size
            of the output layer of the DefectCNN model. Default is 6.
        batch_size (int, optional):
            Batch size used by the DataLoaders during training and evaluation.
            Default is 32.
        num_epochs (int, optional):
            Number of epochs to train the CNN model. Default is 20.
        learning_rate (float, optional):
            Learning rate used by the Adam optimizer. Default is 1e-3.
        device (str, optional):
            Computation device to use, either "cuda" or "cpu". If None, the
            function automatically selects "cuda" when available, otherwise
            falls back to "cpu". Default is None.
        checkpoint_dir (str, optional):
            Directory where the best model checkpoint (best_cnn.pth) will be
            saved. The directory is created if it does not exist. Default is
            "checkpoints_cnn".
    Output:
        metrics (Dict):
            Dictionary containing summary metrics for the trained model, with
            at least the following keys:
                "best_val_accuracy": best validation accuracy achieved during
                                     training (float).
                "test_accuracy":     final accuracy on the test set (float).
                "best_checkpoint":   filesystem path to the saved best model
                                     checkpoint (str).
"""
def train_and_evaluate_cnn(
    data_root: str = "data/cnn",
    num_classes: int = 6,
    batch_size: int = 32,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    device: str = None,
    checkpoint_dir: str = "checkpoints_cnn"
) -> Dict:

    # Set seeds for reproducibility
    set_global_seed(42)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device in used: ", device)

    # Define image transformations for train/val/test
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    # Create train/val/test datasets and loaders
    train_dataset = PCBCNNDataset(root_dir=data_root, split="train",
                                  transform=train_transform)
    val_dataset = PCBCNNDataset(root_dir=data_root, split="val",
                                transform=test_transform)
    test_dataset = PCBCNNDataset(root_dir=data_root, split="test",
                                 transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, optimizer
    model = DefectCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(checkpoint_dir, "best_cnn.pth")

    # Record CNN classifier result data
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        loop = tqdm(train_loader, desc=f"[CNN] Epoch {epoch}/{num_epochs}", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training stats
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation
        val_acc = _evaluate_cnn(model, val_loader, device=device)

        print(f"[CNN] Epoch {epoch:03d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best checkpoint based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[CNN] Saved new best model to {best_ckpt_path}")

    # Load best model for final test evaluation
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_acc = _evaluate_cnn(model, test_loader, device=device)
    print(f"[CNN] Final Test Accuracy: {test_acc:.4f}")

    metrics = {
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "best_checkpoint": best_ckpt_path,
    }

    # Store result in plot for further analysis
    prefix = "deeppcb_cnn" if "deeppcb" in checkpoint_dir.lower() else "kaggle_cnn"
    plot_cnn_history(history, out_dir=checkpoint_dir, prefix=prefix)

    return metrics
"""
    Function Name: _evaluate_cnn
    Description:
        Evaluates a trained CNN model on a provided DataLoader and computes
        classification accuracy. The function switches the model to evaluation
        mode, disables gradient computation, and iterates over all batches in
        the dataloader, accumulating the number of correct predictions and the
        total number of samples. It then returns the accuracy as the ratio of
        correct predictions to the total number of examples.
    Input:
        model (nn.Module):
            Trained CNN model (typically an instance of DefectCNN) to be
            evaluated.
        dataloader (DataLoader):
            PyTorch DataLoader providing the evaluation data (validation or
            test set) as (image, label) batches.
        device (str, optional):
            Computation device used for inference, either "cuda" or "cpu".
            Default is "cpu".
    Output:
        accuracy (float):
            Classification accuracy over all samples in the dataloader. If the
            dataloader is empty (no samples), the function returns 0.0.
"""
def _evaluate_cnn(model: nn.Module,
                  dataloader: DataLoader,
                  device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def plot_cnn_history(history: dict, out_dir: str, prefix: str = "cnn"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(out_dir, f"{prefix}_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    # Loss plot (train loss only, since we’re not tracking val loss)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN training loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, f"{prefix}_loss.png")
    plt.savefig(loss_path)
    plt.close()

    print(f"[CNN] Saved accuracy plot to: {acc_path}")
    print(f"[CNN] Saved loss plot to: {loss_path}")