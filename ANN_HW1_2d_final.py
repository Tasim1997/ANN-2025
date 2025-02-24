import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

#################################
# 1) Load Data
#################################
X_train_df = pd.read_csv("x_training.csv", index_col=0)
y_train_df = pd.read_csv("y_training.csv", index_col=0)

X_val_df   = pd.read_csv("x_val.csv", index_col=0)
y_val_df   = pd.read_csv("y_val.csv", index_col=0)

X_test_df  = pd.read_csv("x_testing.csv", index_col=0)
y_test_df  = pd.read_csv("y_testing.csv", index_col=0)

# Print heads to verify
print("X_train_df:\n", X_train_df.head())
print("y_train_df:\n", y_train_df.head())
print("X_val_df:\n", X_val_df.head())
print("y_val_df:\n", y_val_df.head())
print("X_test_df:\n", X_test_df.head())
print("y_test_df:\n", y_test_df.head())

#################################
# 2) Convert to NumPy + Scale
#################################
X_train_raw = X_train_df.values.astype('float32')
y_train_raw = y_train_df.values.reshape(-1, 1).astype('float32')

X_val_raw = X_val_df.values.astype('float32')
y_val_raw = y_val_df.values.reshape(-1, 1).astype('float32')

X_test_raw = X_test_df.values.astype('float32')
y_test_raw = y_test_df.values.reshape(-1, 1).astype('float32')

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
y_train = scaler_y.fit_transform(y_train_raw)

X_val = scaler_X.transform(X_val_raw)
y_val = scaler_y.transform(y_val_raw)

X_test = scaler_X.transform(X_test_raw)
y_test = scaler_y.transform(y_test_raw)

y_std = scaler_y.scale_[0]  # needed to convert scaled MSE -> original scale

#################################
# 3) Define 3 LR Schedules
#################################

# (a) ReduceLROnPlateau
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7
)

# (b) OneCycleLR (custom)
def one_cycle_lr(epoch, max_lr=1e-3, total_epochs=500, start_lr=5e-5, final_lr=1e-5):
    """
    OneCycleLR: 
      - warmup from start_lr -> max_lr over first 30% of epochs
      - decay from max_lr -> final_lr over next 50%
      - final annealing (last 20%) remains at final_lr
    """
    warmup_epochs = int(total_epochs * 0.3)  # 30%
    decay_epochs  = int(total_epochs * 0.5)  # 50%
    # final 20% is total_epochs - warmup_epochs - decay_epochs

    if epoch < warmup_epochs:
        # Warmup
        lr = start_lr + (max_lr - start_lr) * (epoch / warmup_epochs)
    elif epoch < warmup_epochs + decay_epochs:
        # Decay
        progress = (epoch - warmup_epochs) / decay_epochs
        lr = max_lr - (max_lr - final_lr) * progress
    else:
        # Final 20% is just final_lr
        lr = final_lr
    return lr

onecycle_lr_cb = keras.callbacks.LearningRateScheduler(
    lambda epoch: one_cycle_lr(epoch, max_lr=1e-3, total_epochs=500, start_lr=5e-5, final_lr=1e-5)
)

# (c) CyclicLR (triangular2)
def cyclic_lr(epoch, base_lr=5e-5, max_lr=1e-3, step_size=5, mode='triangular2'):
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    if mode == 'triangular':
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    elif mode == 'triangular2':
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
    else:
        raise ValueError("Mode not recognized. Use 'triangular' or 'triangular2'.")
    return lr

cyclic_lr_cb = keras.callbacks.LearningRateScheduler(
    lambda epoch: cyclic_lr(epoch, base_lr=5e-5, max_lr=1e-3, step_size=5, mode='triangular2')
)

#################################
# 4) Build Model Function
#################################
def build_model():
    input_dim = X_train.shape[1]
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)  # single output
    ])
    # We'll use Adam with a base LR of 5e-5
    opt = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(loss='mse', optimizer=opt)
    return model

#################################
# 5) Train 3 Models 
#################################
EPOCHS = 250

schedules = {
    'ReduceLROnPlateau': [reduce_lr_cb],
    'OneCycleLR': [onecycle_lr_cb],
    'CyclicLR': [cyclic_lr_cb]
}

histories = {}
test_mses_scaled = {}
test_mses_orig   = {}

plt.figure(figsize=(15, 4))

for i, (sched_name, callbacks_list) in enumerate(schedules.items(), start=1):
    print(f"\n=== Training with {sched_name} ===")
    
    model = build_model()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    histories[sched_name] = history
    
    # Evaluate
    test_mse_scaled = model.evaluate(X_test, y_test, verbose=0)
    test_mse_original = test_mse_scaled * (y_std**2)
    
    test_mses_scaled[sched_name] = test_mse_scaled
    test_mses_orig[sched_name]   = test_mse_original
    
    print(f"[{sched_name}] Test MSE (scaled):   {test_mse_scaled:.4f}")
    print(f"[{sched_name}] Test MSE (original): {test_mse_original:.2f}")
    
    # Plot training vs. validation
    plt.subplot(1, 3, i)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(
        f"{sched_name}\nScaled={test_mse_scaled:.3f}, Orig={test_mse_original:.2f}"
    )
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

plt.tight_layout()
plt.show()
