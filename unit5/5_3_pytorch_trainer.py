import torch
import torch.nn.functional as F
from shared_utils import get_dataset_loaders, PyTorchMLP, compute_accuracy_torchmetrics


def compute_total_loss(model, dataloader, device=None):
    if device is None:
        device = torch.device("cpu")
    model = model.eval()

    total_loss = 0.0
    total_samples = 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

    return total_loss / total_samples


def train(model, train_loader, val_loader, optimizer, device=None):
    if device is None:
        device = torch.device("cpu")

    model = model.train()
    total_loss = 0.0
    total_samples = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += labels.size(0)

    avg_train_loss = total_loss / total_samples

    # Validate the model
    val_loss = compute_total_loss(model, val_loader, device=device)

    train_acc = compute_accuracy_torchmetrics(model, train_loader, device=device, num_classes=10)
    val_acc = compute_accuracy_torchmetrics(model, val_loader, device=device, num_classes=10)

    return avg_train_loss, val_loss, train_acc, val_acc


if __name__ == "__main__":
    model = PyTorchMLP(num_features=28 * 28, num_classes=10)
    N_EPOCHS = 10
    LR = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the correct device
    train_loader, val_loader, test_loader = get_dataset_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        train_loss, val_loss, train_acc, val_acc = train(model, train_loader, val_loader, optimizer, device=device)
        print(f"Epoch {epoch + 1}/{N_EPOCHS}:")
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        print(f"Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}")

    print("Testing the model on the test set...")
    test_acc = compute_accuracy_torchmetrics(model, test_loader, device=device, num_classes=10)
    print(f"Test accuracy: {test_acc:.4f}")
