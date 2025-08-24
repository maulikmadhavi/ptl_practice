from lightning import Trainer, LightningModule
from shared_utils import PyTorchMLP, get_dataset_loaders, compute_accuracy
import torch
import torch.nn.functional as F


class LightningModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = PyTorchMLP(num_features=28 * 28, num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataset_loaders()

    lightning_model = LightningModel()
    trainer = Trainer(max_epochs=10, accelerator="auto", devices=1 if torch.cuda.is_available() else None)

    trainer.fit(lightning_model, train_loader, val_loader)

    test_accuracy = compute_accuracy(lightning_model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}")
