import pandas as pd
import numpy as np

# 1) Load Local Data

X_train_df = pd.read_csv("x_training.csv", index_col=0)
y_train_df = pd.read_csv("y_training.csv", index_col=0)
X_test_df  = pd.read_csv("x_testing.csv", index_col=0)
y_test_df  = pd.read_csv("y_testing.csv", index_col=0)

# Print to verify
print("X_train_df:\n", X_train_df.head())
print("y_train_df:\n", y_train_df.head())
print("X_test_df:\n", X_test_df.head())
print("y_test_df:\n", y_test_df.head())


# 6. Convert to NumPy arrays
X_train = np.array(X_train_df)
y_train = np.array(y_train_df)
X_test  = np.array(X_test_df)
y_test  = np.array(y_test_df)

# 7. Append a dummy feature (column of 1's) to learn an intercept
ones_train = np.ones((X_train.shape[0], 1))
ones_test  = np.ones((X_test.shape[0], 1))

X_train = np.hstack((ones_train, X_train))
X_test  = np.hstack((ones_test, X_test))

# 8. Solve for w using the closed-form equation: w = (Xᵀ X)⁻¹ Xᵀ y

y_train = y_train.reshape(-1, 1)

# Compute (Xᵀ X) and (Xᵀ y)
XTX = X_train.T @ X_train
XTy = X_train.T @ y_train

# Invert (Xᵀ X) and multiply
w = np.linalg.inv(XTX) @ XTy

# 9. Predict on training and testing sets
y_train_pred = X_train @ w
y_test_pred  = X_test  @ w

# 10. Compute Mean Squared Error (MSE)
mse_train = np.mean((y_train_pred - y_train)**2)
mse_test = 0
if y_test.size > 0:
    # Only compute if the test set is not empty
    y_test = y_test.reshape(-1, 1)  # ensure shape is [num_samples, 1]
    mse_test = np.mean((y_test_pred - y_test)**2)

# 11. Print results
print("===== Linear Regression Results =====")
print(f"Coefficients (w): \n{w}\n")
print("\n=== Testing Predictions ===")
#print(y_test_pred)
print(f"Training MSE : {mse_train}")
print(f"Testing  MSE : {mse_test}")
