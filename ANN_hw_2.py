import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ----------------
# 1) Dataset
# ----------------
class StockDataset(Dataset):
    def __init__(self, X, Y, days):
        """
        X shape: (num_features, num_days)
        Y shape: (num_days, )
        days: number of days used as input window
        """
        self.X = X
        self.Y = Y.reshape(-1)
        self.days = days

    def __len__(self):
        return len(self.Y) - self.days

    def __getitem__(self, index):
       
        x = self.X[:, index:index+self.days]
        y = self.Y[index + self.days]
        return x, y

def get_price(tick, start='2020-01-01', end=None):
    return yf.Ticker(tick).history(start=start, end=end)['Close']

def get_prices(tickers, start='2020-01-01', end=None):
    df = pd.DataFrame()
    for s in tickers:
        df[s] = get_price(s, start, end)
    return df

feature_stocks = [
    'tsla','meta','nvda','amzn','nflx','gbtc','gdx','intc','dal',
    'c','goog','aapl','msft','ibm','hp','orcl','sap','crm','hubs','twlo'
]
predict_stock  = 'msft'
start_date     = '2020-01-01'

allX = get_prices(feature_stocks, start=start_date)
ally = get_prices([predict_stock], start=start_date)

# Create Dataset using the last 5 days
days = 5
stockData = StockDataset(
    allX.to_numpy().transpose().astype(np.float32),
    ally.to_numpy().astype(np.float32),
    days=days
)

# ----------------
# 2) Train / Valid / Test Split
# ----------------
total_size = len(stockData)
train_set_size = int(total_size * 0.7)
valid_set_size = int(total_size * 0.15)
test_set_size = total_size - train_set_size - valid_set_size

train_set, valid_set, test_set = random_split(
    stockData,
    [train_set_size, valid_set_size, test_set_size],
    generator=torch.Generator().manual_seed(42)
)

# ----------------
# 3) MLP Model
# ----------------
class MLPForecast(pl.LightningModule):
    def __init__(self, input_dim=100, hidden_dim=160, dropout_rate=0.2, lr=1e-4):
        """
        input_dim = 20 features * 5 days = 100
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
       
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# ----------------
# 4) DataModule
# ----------------
class StockDataModule(pl.LightningDataModule):
    def __init__(self, train_set, valid_set, test_set, batch_size=32):
        super().__init__()
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set  = test_set
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
        self.train_set,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=15,
        persistent_workers=True  # Keeps workers alive across epochs
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=15,
            persistent_workers=True
        )
        

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=15,
            persistent_workers=True
        )
            

batch_size = 128
data_module = StockDataModule(train_set, valid_set, test_set, batch_size=batch_size)

# Instantiate the model
model = MLPForecast(
    input_dim=days * len(feature_stocks),
    hidden_dim=200,
    dropout_rate=0.2,
    lr=1e-3
)

# ----------------
# 5) Callbacks: LossHistoryLogger and LitModelCheckpoint
# ----------------
class LossHistoryLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.item())

from pytorch_lightning.callbacks import ModelCheckpoint

# Custom checkpoint callback simulating seamless uploading to a model registry.
class LitModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        # Simulate uploading the checkpoint to a model registry.
        print("Uploading checkpoint to model registry...")

lit_checkpoint_callback = LitModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    mode="min",
    filename="best-checkpoint"
)

loss_history_callback = LossHistoryLogger()

# ----------------
# 6) Main Entry Point
# ----------------
if __name__ == '__main__':
    trainer = pl.Trainer(
        max_epochs=70,
        deterministic=True,
        log_every_n_steps=1,
        callbacks=[loss_history_callback, lit_checkpoint_callback]
    )

    trainer.fit(model, data_module)
    test_results = trainer.test(model, datamodule=data_module)
    print("Test Results:", test_results)


    # 7) Plot the Training and Validation MSE
   
    if len(loss_history_callback.val_losses) > len(loss_history_callback.train_losses):
        val_losses = loss_history_callback.val_losses[1:]
    else:
        val_losses = loss_history_callback.val_losses

    epochs = range(len(loss_history_callback.train_losses))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history_callback.train_losses, label='Train MSE')
    plt.plot(epochs, val_losses, label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.legend()
    plt.grid(True)
    plt.show()
