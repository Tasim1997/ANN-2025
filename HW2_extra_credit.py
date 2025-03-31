
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import optuna  # pip install optuna

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

# ----------------
# 3) Lookback Window, Split Data
# ----------------
days = 20  # 20-day lookback does NOT change the model's parameter count.
stockData = StockDataset(allX_scaled, ally_scaled, days=days)

total_size = len(stockData)
train_size = int(total_size * 0.7)
val_size   = int(total_size * 0.15)
test_size  = total_size - train_size - val_size

train_set, val_set, test_set = random_split(
    stockData, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# ----------------
# 4) CNN Model (architecture unchanged => same trainable params)
# ----------------
class CNNForecast(pl.LightningModule):
    def __init__(self, in_channels=20, seq_len=20, dropout_rate=0.3, lr=1e-3, weight_decay=1e-5):
        """
        in_channels: number of stock features
        seq_len: number of days
        dropout_rate, lr, weight_decay are hyperparameters
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Add a ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# ----------------
# 5) LightningDataModule
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
# 6) Optuna Objective: tune dropout_rate, lr, weight_decay, etc.
# ----------------
def objective(trial):
    """
    The objective function for Optuna. It will:
    1) Suggest hyperparameters.
    2) Create a model.
    3) Train the model for a small number of epochs.
    4) Return the validation loss.
    """
    import math
    
    # Suggest hyperparams
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.7)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    
    # Create model with these hyperparameters
    model = CNNForecast(
        in_channels=20,
        seq_len=20,
        dropout_rate=dropout_rate,
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Quick trainer for hyperparam search
    trainer = pl.Trainer(
        max_epochs=20,                # fewer epochs for trial evaluation
        enable_checkpointing=False,   # do not save checkpoints
        logger=False,                 # disable logging
        deterministic=True
    )
    
    trainer.fit(model, data_module)
    
    # Retrieve the final validation loss
    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        return float("inf")
    return val_loss.item()

# ----------------
# 7) Main: Optuna Tuning, Then Final 250-Epoch Training
# ----------------
if __name__ == '__main__':
    # 1) Create the Optuna study
    study = optuna.create_study(direction="minimize")
    
    # 2) Optimize (search) the hyperparameters
    study.optimize(objective, n_trials=10)  # Increase n_trials if needed
    
    # 3) Retrieve best hyperparams
    best_params = study.best_params
    print("Best hyperparameters from Optuna:", best_params)
    
    # 4) Create a final model with the best hyperparams
    final_model = CNNForecast(
        in_channels=20,
        seq_len=20,
        dropout_rate=best_params["dropout_rate"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"]
    )
    
    # 5) Train for 250 epochs
    final_trainer = pl.Trainer(
        max_epochs=250,
        deterministic=True,
        log_every_n_steps=1
    )
    final_trainer.fit(final_model, data_module)
    
    # 6) Evaluate on the test set
    test_results = final_trainer.test(final_model, datamodule=data_module)
    print("Test Results (Normalized Scale):", test_results)
    
    # 7) Save the final model as extra_credit
    final_trainer.save_checkpoint("extra_credit.ckpt")
    torch.save(final_model.state_dict(), "extra_credit.h5")
    
    # 8) Predict on the test set in real scale
    test_loader = data_module.test_dataloader()
    all_predictions = []
    all_targets = []
    final_model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            preds = final_model(x)
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions).reshape(-1, 1)
    all_targets = np.concatenate(all_targets).reshape(-1, 1)
    
    # Inverse transform
    predictions_real = scalerY.inverse_transform(all_predictions)
    targets_real = scalerY.inverse_transform(all_targets)
    
    # MSE on original scale
    mse_real = mean_squared_error(targets_real, predictions_real)
    print("MSE on original scale:", mse_real)
    
    # 9) Print total trainable parameters (should remain ~39,361)
    total_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print("Final Total Trainable Parameters:", total_params)
    
    # Skipping any plotting, as requested.
