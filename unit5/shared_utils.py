from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from collections import Counter
import random
from torchmetrics import Accuracy
from lightning import LightningDataModule

torch.random.manual_seed(42)
random.seed(42)


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(25, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.all_layers(x)


def get_dataset_loaders():
    train_dataset = datasets.MNIST(root="./mnist", train=True, transform=transforms.ToTensor(), download=True)

    test_dataset = datasets.MNIST(root="./mnist", train=False, transform=transforms.ToTensor())

    train_dataset, val_dataset = random_split(
        train_dataset, lengths=[55000, 5000], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def compute_accuracy(model, dataloader, device=None):
    if device is None:
        device = torch.device("cpu")
    model = model.eval()

    correct = 0.0
    total_examples = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


def compute_accuracy_torchmetrics(model, dataloader, device=None, num_classes=10):
    if device is None:
        device = torch.device("cpu")

    model = model.eval()
    accuracy = Accuracy(num_classes=num_classes, task="multiclass").to(device)

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        accuracy(predictions, labels)

    return accuracy.compute()


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        datasets.MNIST(root="./mnist", train=True, download=True)
        datasets.MNIST(root="./mnist", train=False, download=True)

    def setup(self, stage=None):
        self.mnist_test = datasets.MNIST(root="./mnist", train=False, transform=transforms.ToTensor())
        self.mnist_predict = datasets.MNIST(root="./mnist", train=False, transform=transforms.ToTensor())
        self.mnist_trainval = datasets.MNIST(root="./mnist", train=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(
            self.mnist_trainval, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, shuffle=False, num_workers=0)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataset_loaders()

    # Get the stats for train_loader, val_loader, test_loader
    train_counter = Counter()
    test_counter = Counter()
    val_counter = Counter()
    for data in train_loader:
        labels = data[1]
        train_counter.update(labels.tolist())

    for data in val_loader:
        labels = data[1]
        val_counter.update(labels.tolist())

    for data in test_loader:
        labels = data[1]
        test_counter.update(labels.tolist())

    print(f"train_counter: {sorted(train_counter.items())}")
    print(f"val_counter: {sorted(val_counter.items())}")
    print(f"test_counter: {sorted(test_counter.items())}")

    # zero _ rule classification/select the class with the highest training frequency
    zero_rule_class = train_counter.most_common(1)[0][0]
    print(
        f"Zero-rule classifier predicts: {zero_rule_class}, with probability: {train_counter[zero_rule_class] / sum(train_counter.values()): .4f}"
    )
