from pyexpat import features
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
        self.accuracy_test = Accuracy(num_classes=10, task="multiclass")

    def forward(self, x):
        return self.model(x)

    # NEW !!!
    # Shared step that is used in
    # training_step, validation_step, and tst_step
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)

        self.accuracy_train(predicted_labels, true_labels)
        self.log("train_acc", self.accuracy_train.compute(), prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)

        self.accuracy_val(predicted_labels, true_labels)
        self.log("val_acc", self.accuracy_val.compute(), prog_bar=True)
        return loss

    # NEW !!!
    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.accuracy_test(predicted_labels, true_labels)
        self.log("accuracy", self.accuracy_test.compute())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataset_loaders()
    torch.manual_seed(42)
    lightning_model = LightningModel(learning_rate=0.05)
    trainer = Trainer(
        max_epochs=10, accelerator="auto", devices=1 if torch.cuda.is_available() else None, deterministic=True
    )

    trainer.fit(lightning_model, train_loader, val_loader)

    # Evaluate model based on test_step
    train_acc = trainer.test(dataloaders=train_loader)[0]["accuracy"]
    val_acc = trainer.test(dataloaders=val_loader)[0]["accuracy"]
    test_acc = trainer.test(dataloaders=test_loader)[0]["accuracy"]
    print(f"Train Acc {train_acc * 100:.2f}% | Val Acc {val_acc * 100:.2f}% | Test Acc {test_acc * 100:.2f}%")
# Train Acc 97.54% | Val Acc 97.45% | Test Acc 97.29%
