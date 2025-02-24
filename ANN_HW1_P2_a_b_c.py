import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

#################################
# 1) Load Pre-Split Data from CSV Files
#################################
X_train_df = pd.read_csv("x_training.csv", index_col=0)
y_train_df = pd.read_csv("y_training.csv", index_col=0)

X_val_df   = pd.read_csv("x_val.csv", index_col=0)
y_val_df   = pd.read_csv("y_val.csv", index_col=0)

X_test_df  = pd.read_csv("x_testing.csv", index_col=0)
y_test_df  = pd.read_csv("y_testing.csv", index_col=0)

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

#################################
# 4) Build Model (5-Layer, 20 Neurons Each, ReLU)
#################################
def build_model(optimizer):
    input_dim = X_train.shape[1]
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)  # single output for regression
    ])
    model.compile(loss='mse', optimizer=optimizer)
    return model

#################################
# 5) Define Optimizers
#################################
optimizers = {
    'SGD': keras.optimizers.SGD(learning_rate=5e-5),
    'Momentum': keras.optimizers.SGD(learning_rate=5e-5, momentum=0.9),
    'Adam': keras.optimizers.Adam(learning_rate=5e-5)
}

#################################
# 6) Train & Evaluate
#################################
histories = {}
test_mses_scaled = {}
test_mses_original = {}

plt.figure(figsize=(12, 4))

# We'll need the std of y to convert scaled MSE -> original scale
y_std = scaler_y.scale_[0]  # the standard deviation from scaler_y

for i, (opt_name, opt_obj) in enumerate(optimizers.items(), start=1):
    print(f"\n=== Training with {opt_name} ===")
    
    # Build a fresh model for each optimizer
    model = build_model(optimizer=opt_obj)
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=250,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    histories[opt_name] = history
    
    # Evaluate on test set (scaled MSE)
    test_mse_scaled = model.evaluate(X_test, y_test, verbose=0)
    # Convert to original scale MSE
    test_mse_orig = test_mse_scaled * (y_std ** 2)
    
    test_mses_scaled[opt_name] = test_mse_scaled
    test_mses_original[opt_name] = test_mse_orig
    
    print(f"Test MSE (scaled) using {opt_name}: {test_mse_scaled:.2f}")
    print(f"Test MSE (original) using {opt_name}: {test_mse_orig:.2f}")
    
    # Plot training vs validation loss
    plt.subplot(1, 3, i)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    # Show scaled & original MSE in the plot title
    plt.title(
        f"{opt_name}\nTrain vs. Val Loss\n"
        f"(Scaled={test_mse_scaled:.3f}, Orig={test_mse_orig:.2f})"
    )
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

plt.tight_layout()
plt.show()
