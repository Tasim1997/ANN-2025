import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler

import yfinance as yf
import pandas as pd

def get_price(tick, start='2022-10-01', end=None):
    """
    Download historical Close prices for a single ticker using yfinance.
    """
    return yf.Ticker(tick).history(start=start, end=end)['Close']

def get_prices(tickers, start='2022-10-01', end=None):

    df = pd.DataFrame()
    for s in tickers:
        df[s] = get_price(s, start, end)
    return df

# Define your feature tickers and the stock to be predicted
feature_stocks = ['tsla','meta','goog','amzn','nflx','gbtc','gdx','intc','dal','c']
predict_stock  = 'msft'



train_start = '2023-01-01'
train_end   = '2024-06-30'
val_start   = '2024-07-01'
val_end     = '2024-10-31'
test_start  = '2024-11-01'
test_end    = '2024-12-31'

# -- Download Data from yfinance --
X_train_df = get_prices(feature_stocks, start=train_start, end=train_end)
y_train_df = get_prices([predict_stock],  start=train_start, end=train_end)

X_val_df   = get_prices(feature_stocks, start=val_start, end=val_end)
y_val_df   = get_prices([predict_stock],  start=val_start, end=val_end)

X_test_df  = get_prices(feature_stocks, start=test_start, end=test_end)
y_test_df  = get_prices([predict_stock],  start=test_start, end=test_end)

# Print to verify
print("X_train_df head:")
print(X_train_df.head())
print("\ny_train_df head:")
print(y_train_df.head())
print("\nX_val_df head:")
print(X_val_df.head())
print("\ny_val_df head:")
print(y_val_df.head())
print("\nX_test_df head:")
print(X_test_df.head())
print("\ny_test_df head:")
print(y_test_df.head())


#################################
# 2) Convert DataFrames to NumPy Arrays
#################################
X_train_raw = X_train_df.values.astype('float32')
y_train_raw = y_train_df.values.reshape(-1, 1).astype('float32')

X_val_raw = X_val_df.values.astype('float32')
y_val_raw = y_val_df.values.reshape(-1, 1).astype('float32')

X_test_raw = X_test_df.values.astype('float32')
y_test_raw = y_test_df.values.reshape(-1, 1).astype('float32')

#################################
# 3) Scale Data
#################################
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
y_train = scaler_y.fit_transform(y_train_raw)

X_val = scaler_X.transform(X_val_raw)
y_val = scaler_y.transform(y_val_raw)

X_test = scaler_X.transform(X_test_raw)
y_test = scaler_y.transform(y_test_raw)

# We'll need this for converting scaled MSE -> original scale
y_std = scaler_y.scale_[0]

#################################
# 4) Triangular2 CycLR
#################################
def cyc_lr_triangular2(epoch, base_lr=1e-5, max_lr=1e-2, step_size=10):
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
    return lr

lr_scheduler = keras.callbacks.LearningRateScheduler(
    lambda epoch: cyc_lr_triangular2(epoch, base_lr=1e-5, max_lr=1e-2, step_size=10)
)

#################################
# 5) Build a 5x18 Model with BN & Dropout
#################################
def build_model():
    input_dim = X_train.shape[1]
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Hidden Layer 1
        layers.Dense(18, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden Layer 2
        layers.Dense(18, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden Layer 3
        layers.Dense(18, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden Layer 4
        layers.Dense(18, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden Layer 5
        layers.Dense(18, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Output
        layers.Dense(1)
    ])
    # Use Adam
    opt = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss='mse', optimizer=opt)
    return model

#################################
# 6) Train the Model With CyclicLR Scheduler
#################################
model = build_model()
history = model.fit(
    X_train, y_train,
    epochs=250,  # changed to 200
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler],
    verbose=1
)

#################################
# 7) Evaluate the Model (Scaled + Original MSE)
#################################
test_mse_scaled = model.evaluate(X_test, y_test, verbose=0)
test_mse_original = test_mse_scaled * (y_std ** 2)
print(f"Test MSE (scaled):   {test_mse_scaled:.4f}")
print(f"Test MSE (original): {test_mse_original:.2f}")

#################################
# 8) Display Parameter Count
#################################
model.summary()  # Verify total parameters

#################################
# 9) Plot Training vs. Validation Loss
#################################
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title(f"Train vs. Val Loss\n(Scaled={test_mse_scaled:.4f}, Orig={test_mse_original:.2f})")
plt.show()

#################################
# 10) Show Predictions in Original Scale
#################################
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = y_test_df.values

print("\nSample predictions vs. actual:")
for i in range(min(5, len(y_true))):
    print(f"Predicted: {y_pred[i][0]:.2f} | Actual: {y_true[i][0]:.2f}")

#################################
# 11) Save the Model (Coefficients)
#################################
model.save("my_model2.h5")
print("\nModel saved to 'my_model.h5' with all weights (coefficients).")
