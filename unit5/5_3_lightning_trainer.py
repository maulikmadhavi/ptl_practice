from lightning import Trainer, LightningModule
from shared_utils import PyTorchMLP, get_dataset_loaders, compute_accuracy_torchmetrics
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy


class LightningModel(LightningModule):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.model = PyTorchMLP(num_features=28 * 28, num_classes=10)
        self.learning_rate = learning_rate
        self.accuracy_train = Accuracy(num_classes=10, task="multiclass")
        self.accuracy_val = Accuracy(num_classes=10, task="multiclass")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        pred = torch.argmax(logits, dim=1)

        self.accuracy_train(pred, labels)
        self.log("train_acc", self.accuracy_train.compute(), prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        pred = torch.argmax(logits, dim=1)
        self.accuracy_val(pred, labels)
        self.log("val_acc", self.accuracy_val.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataset_loaders()

    lightning_model = LightningModel(learning_rate=0.05)
    trainer = Trainer(max_epochs=10, accelerator="auto", devices=1 if torch.cuda.is_available() else None)

    trainer.fit(lightning_model, train_loader, val_loader)

    test_accuracy = compute_accuracy_torchmetrics(lightning_model, test_loader, num_classes=10)
    train_accuracy = compute_accuracy_torchmetrics(lightning_model, train_loader, num_classes=10)
    val_accuracy = compute_accuracy_torchmetrics(lightning_model, val_loader, num_classes=10)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}")
