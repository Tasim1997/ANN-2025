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
# Load feature (X) and target (y) data from CSV files.
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
# 4) Build a 5-Layer Network (20 neurons each, ReLU)
#################################
def build_model():
    input_dim = X_train.shape[1]
    model = keras.Sequential([
        layers.Dense(20, activation='relu', input_shape=(input_dim,)),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)  # single output for regression
    ])

    # Use RMSprop as the optimizer
    opt = keras.optimizers.RMSprop(learning_rate=1e-5)
    model.compile(loss='mse', optimizer=opt)
    return model

#################################
# 5) Train the Model
#################################
model = build_model()
history = model.fit(
    X_train, y_train,
    epochs=500,  
    validation_data=(X_val, y_val),
    verbose=1
)

#################################
# 6) Evaluate on Test Set
#################################
test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_mse:.4f}")

#################################
# 7) Plot Training vs. Validation Loss
#################################
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title(f"5-Layer MLP (20 neurons, RMSprop)\nTest MSE={test_mse:.4f}")
plt.show()

#################################
# 8) (Optional) Show Predictions in Original Scale
#################################
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = y_test_df.values

print("\nSample predictions vs. actual:")
for i in range(min(5, len(y_true))):
    print(f"Predicted: {y_pred[i][0]:.2f} | Actual: {y_true[i][0]:.2f}")
# Evaluate in scaled form:
test_mse_scaled = model.evaluate(X_test, y_test, verbose=0)
# Convert MSE back to original scale:
y_std = scaler_y.scale_[0]      # the standard deviation used by scaler_y
test_mse_original_scale = test_mse_scaled * (y_std**2)
print("Test MSE in original price scale:", test_mse_original_scale)
