import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim as optim
from torchmetrics import MeanSquaredError

class RegressionModel(LightningModule):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], lr=0.0001, dropout_rate=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model(input_dim, hidden_dims, dropout_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self, input_dim, hidden_dims, dropout_rate):
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = MeanSquaredError()(y_hat, y)
        loss = self.criterion(y_hat, y)
        metrics = {'test_acc': mse, 'test_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
