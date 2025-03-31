import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ----------------
# 1) Data Download and Normalization
# ----------------
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
predict_stock = 'msft'
start_date = '2020-01-01'
end_date = None  # up-to-date data

# Download data
allX = get_prices(feature_stocks, start=start_date, end=end_date)
ally = get_prices([predict_stock], start=start_date, end=end_date)

# Fill missing values if any
allX = allX.fillna(method='ffill')
ally = ally.fillna(method='ffill')

# Normalize data using StandardScaler
scalerX = StandardScaler()
scalerY = StandardScaler()

# Fit scalers on the data (features and target)
allX_scaled = pd.DataFrame(scalerX.fit_transform(allX), index=allX.index, columns=allX.columns)
ally_scaled = pd.DataFrame(scalerY.fit_transform(ally), index=ally.index, columns=ally.columns)

# ----------------
# 2) Dataset
# ----------------
class StockDataset(Dataset):
    def __init__(self, X, Y, days):
        """
        X: DataFrame of shape (num_days, num_features)
        Y: DataFrame of shape (num_days, 1)
        days: number of days used as input window
        """
        # Transpose X to have shape (num_features, num_days)
        self.X = X.to_numpy().T.astype(np.float32)
        self.Y = Y.to_numpy().astype(np.float32).reshape(-1)
        self.days = days

    def __len__(self):
        return len(self.Y) - self.days

    def __getitem__(self, index):
        x = self.X[:, index:index+self.days]  # (num_features, days)
        y = self.Y[index + self.days]
        return x, y

days = 5  # try a larger lookback window, e.g., 15–30, if desired
stockData = StockDataset(allX_scaled, ally_scaled, days=days)

# ----------------
# 3) Train/Validation/Test Split
# ----------------
total_size = len(stockData)
train_size = int(total_size * 0.7)
val_size   = int(total_size * 0.15)
test_size  = total_size - train_size - val_size

train_set, val_set, test_set = random_split(
    stockData, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# ----------------
# 4) CNN Model
# ----------------
class CNNForecast(pl.LightningModule):
    def __init__(self, in_channels=20, seq_len=5, dropout_rate=0.2, lr=1e-3):
        """
        in_channels: number of stock features (default=20)
        seq_len: number of days (default=5)
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.criterion = nn.MSELoss()
        
        # First convolution block
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        
        # Second convolution block
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Global Pool
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Extra fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        
        # Block 1
        x = self.conv1(x)         # -> (batch_size, 64, seq_len)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        # Block 2
        x = self.conv2(x)         # -> (batch_size, 128, seq_len)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.dropout(x)
        
        # Pool (adaptive avg across time dimension)
        x = self.pool(x)          # -> (batch_size, 128, 1)
        x = x.squeeze(-1)         # -> (batch_size, 128)
        
        # Fully connected layers
        x = self.leaky_relu(self.fc1(x))   # -> (batch_size, 64)
        x = self.leaky_relu(self.fc2(x))   # -> (batch_size, 32)
        x = self.fc3(x)                    # -> (batch_size, 1)
        
        return x.squeeze(-1)  # -> (batch_size,)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# ----------------
# 5) LightningDataModule for Stock Data
# ----------------
class StockDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, test_ds, batch_size=128):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

batch_size = 128
data_module = StockDataModule(train_set, val_set, test_set, batch_size=batch_size)

# ----------------
# 6) Custom Callback for Logging Train & Validation Loss in Sync
# ----------------
class LossHistoryLogger(pl.Callback):
    """
    Logs train_loss and val_loss only after each validation epoch end,
    ensuring they have the same length when we plot them.
    """
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of every validation epoch."""
        # We'll only log if both train_loss and val_loss exist
        metrics = trainer.callback_metrics
        if "train_loss" in metrics and "val_loss" in metrics:
            train_loss = metrics["train_loss"].item()
            val_loss   = metrics["val_loss"].item()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # current_epoch starts at 0, goes to max_epochs-1
            self.epochs.append(trainer.current_epoch)

# ----------------
# 7) Main Entry Point: Training, Testing, and Plotting
# ----------------
if __name__ == '__main__':
    model = CNNForecast(in_channels=20, seq_len=days, dropout_rate=0.3, lr=1e-3)
    
    # Create an instance of the callback
    loss_history_callback = LossHistoryLogger()
    
    trainer = pl.Trainer(
        max_epochs=100, 
        deterministic=True,
        log_every_n_steps=1,
        callbacks=[loss_history_callback]  # Add our logging callback
    )
    
    # Train and Test
    trainer.fit(model, data_module)
    test_results = trainer.test(model, datamodule=data_module)
    print("Test Results (Normalized Scale):", test_results)
    
    # Save the model checkpoint and weights
    trainer.save_checkpoint("best_model_2conv.ckpt")
    torch.save(model.state_dict(), "best_model_2conv.h5")
    
    # Compute real-scale predictions on the test set.
    test_loader = data_module.test_dataloader()
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions).reshape(-1, 1)
    all_targets = np.concatenate(all_targets).reshape(-1, 1)
    
    # Inverse transform to original scale
    predictions_real = scalerY.inverse_transform(all_predictions)
    targets_real = scalerY.inverse_transform(all_targets)
    
    mse_real = mean_squared_error(targets_real, predictions_real)
    print("MSE on original scale:", mse_real)
    
    # Print the total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Final Total Trainable Parameters:", total_params)
    
    # ----------------
    # Plot Train vs Validation Loss
    # ----------------
    train_losses = loss_history_callback.train_losses
    val_losses   = loss_history_callback.val_losses
    epochs       = loss_history_callback.epochs
    
    print(f"Lengths: epochs={len(epochs)}, train_losses={len(train_losses)}, val_losses={len(val_losses)}")
    
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()
