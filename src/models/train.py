import os
import sys
import joblib
import random
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import DatePreprocessor, SlidingWindowTransformer

# Load params
params = yaml.safe_load(open("params.yaml"))
train_params = params["train"]
preprocess_params = params["preprocess"]

test_size = train_params["test_size"]
window_size = train_params["window_size"]
target_col = train_params["target_col"]
feature_cols = train_params["feature_cols"]
random_state = train_params["random_state"]
data_path = preprocess_params["output_path"]

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

# ─────────────────────────────────────────────
# Load data — only historical rows with known target
# ─────────────────────────────────────────────
df = pd.read_csv(data_path)
df = df[df["is_forecast"] == False].copy()
df = df[df[target_col].notna()].copy()

all_cols = ["Date", target_col] + feature_cols
df = df[all_cols]

# Fill missing dates
date_preprocessor = DatePreprocessor("Date")
df = date_preprocessor.fit_transform(df)
df = df.drop(columns=["Date"])

# ─────────────────────────────────────────────
# Train/test split
# ─────────────────────────────────────────────
df_test = df.iloc[-test_size:]
df_train = df.iloc[:-test_size]

# ─────────────────────────────────────────────
# Separate pipelines for target and features
# ─────────────────────────────────────────────
target_pipeline = Pipeline([
    ("fillna", SimpleImputer(strategy="mean")),
    ("normalize", MinMaxScaler())
])

feature_pipeline = Pipeline([
    ("fillna", SimpleImputer(strategy="mean")),
    ("normalize", MinMaxScaler())
])

preprocess = ColumnTransformer([
    ("target_transformer", target_pipeline, [target_col]),
    ("feature_transformer", feature_pipeline, feature_cols),
])

sliding_window_transformer = SlidingWindowTransformer(window_size)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("sliding_window_transformer", sliding_window_transformer),
])

# ─────────────────────────────────────────────
# Applying pipeline
# ─────────────────────────────────────────────
X_train, y_train = pipeline.fit_transform(df_train)
X_test, y_test = pipeline.transform(df_test)

# only target column (index 0 — energy_demand)
y_train = y_train[:, 0:1]
y_test = y_test[:, 0:1]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Target scaler for inverse transform
target_scaler = pipeline.named_steps["preprocess"].transformers_[0][1].named_steps["normalize"]


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
# todo: can this be done better?
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=60,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# ─────────────────────────────────────────────
# Evaluation on test set
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = target_scaler.inverse_transform(y_pred)

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"Test MAE:  {mae:.4f}")
print(f"Test MSE:  {mse:.4f}")
print(f"Test RMSE: {np.sqrt(mse):.4f}")
# todo: we should add more metrics here and below

# ─────────────────────────────────────────────
# Train on full dataset
# ─────────────────────────────────────────────
X_full, y_full = pipeline.fit_transform(df)

y_full = y_full[:, 0:1]  # only the target column

# Re-get target scaler after re-fitting on full data
target_scaler_full = pipeline.named_steps["preprocess"].transformers_[0][1].named_steps["normalize"]

# Second training — fresh early_stopping
# todo: i can remove this and use the first one in both cases
early_stopping_full = EarlyStopping(
    monitor="val_loss",
    patience=60,
    restore_best_weights=True
)
model = build_model((X_full.shape[1], X_full.shape[2]))
model.fit(
    X_full, y_full,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping_full],
    verbose=1
)

y_pred_full = model.predict(X_full)

y_full_inv = target_scaler_full.inverse_transform(y_full.reshape(-1, 1))
y_pred_full_inv = target_scaler_full.inverse_transform(y_pred_full)

mse_full = mean_squared_error(y_full_inv, y_pred_full_inv)
mae_full = mean_absolute_error(y_full_inv, y_pred_full_inv)
print(f"Full dataset MAE:  {mae_full:.4f}")
print(f"Full dataset MSE:  {mse_full:.4f}")
print(f"Full dataset RMSE: {np.sqrt(mse_full):.4f}")

# ─────────────────────────────────────────────
# Save model and pipeline
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save("models/model_energy_demand.keras")
joblib.dump(pipeline, "models/pipeline_energy_demand.pkl")
print("Model and pipeline saved to models/")
