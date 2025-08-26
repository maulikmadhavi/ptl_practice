from pyexpat import features
from lightning import Trainer, LightningModule
from shared_utils import PyTorchMLP, get_dataset_loaders, MNISTDataModule
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


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
        self.log("train_acc", self.accuracy_train, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)

        self.accuracy_val(predicted_labels, true_labels)
        self.log("val_acc", self.accuracy_val, prog_bar=True, on_epoch=True, on_step=False)

    # NEW !!!
    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.accuracy_test(predicted_labels, true_labels)
        self.log("test_acc", self.accuracy_test, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    torch.manual_seed(42)
    dm = MNISTDataModule()
    lightning_model = LightningModel(learning_rate=0.05)
    YOUR_LOG_ROOT = "logs/"
    tb_logger = TensorBoardLogger(YOUR_LOG_ROOT)
    csv_logger = CSVLogger(YOUR_LOG_ROOT, version=tb_logger.version)
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        deterministic=True,
        default_root_dir=YOUR_LOG_ROOT,
        logger=[tb_logger, csv_logger],
    )

    trainer.fit(lightning_model, datamodule=dm)

    # Evaluate model based on test_step

    train_accuracy = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_acc"]
    test_accuracy = trainer.test(datamodule=dm)[0]["test_acc"]
    val_accuracy = trainer.validate(datamodule=dm)[0]["val_acc"]

    # display
    print(
        f"Train Acc {train_accuracy * 100:.2f}% | Val Acc {val_accuracy * 100:.2f}% | Test Acc {test_accuracy * 100:.2f}%"
    )
