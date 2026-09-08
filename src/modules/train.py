"""
Training loop and accuracy evaluation for DemoGPT.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def calculate_accuracy(model, data_loader, device):
    """
    Calculate the accuracy of the model on the validation dataset.

    Args:
        model (torch.nn.Module): The trained transformer model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model (e.g., 'cuda' or 'cpu').

    Returns:
        float: Validation accuracy as a percentage.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = (total_correct / total_samples) * 100
    return accuracy


def train_model(model, train_loader, val_loader, device, epochs=3, learning_rate=3e-4):
    """
    Trains the model for the given number of epochs, printing running loss
    every 100 steps and validation accuracy at the end of each epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        device (torch.device): Device to train on.
        epochs (int): Number of training epochs.
        learning_rate (float): Optimizer learning rate.

    Returns:
        torch.nn.Module: The trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (input_ids, labels) in enumerate(train_loader):
            # Move data to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log training progress
            if (step + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], "
                      f"Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        # Evaluate validation accuracy
        val_accuracy = calculate_accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.2f}%")

    return model
