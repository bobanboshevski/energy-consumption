from dotenv import load_dotenv
import mlflow
import yaml
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from preprocess import DatePreprocessor, SlidingWindowTransformer
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

load_dotenv()

# ─────────────────────────────────────────────
# Load params
# ─────────────────────────────────────────────
params = yaml.safe_load(open("params.yaml"))
train_params = params["train_univariate"]
preprocess_params = params["preprocess"]

test_size = train_params["test_size"]
window_size = train_params["window_size"]
target_col = train_params["target_col"]
random_state = train_params["random_state"]
data_path = preprocess_params["output_path"]

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)


def build_model(input_shape):
    """
    Univariate LSTM — input is (window_size, 1) since we only use energy_demand.
    Slightly deeper than the multivariate model to compensate for less input information.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


with mlflow.start_run(run_name="train_energy_demand_univariate"):
    mlflow.log_param("model_type", "univariate_lstm")
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("window_size", window_size)
    mlflow.log_param("target_col", target_col)
    mlflow.log_param("random_state", random_state)

    # ─────────────────────────────────────────────
    # Load data — only historical rows with known target
    # ─────────────────────────────────────────────
    df = pd.read_csv(data_path)
    df = df[df["is_forecast"] == False].copy()
    df = df[df[target_col].notna()].copy()
    df = df[["Date", target_col]]

    # Fill missing dates
    date_preprocessor = DatePreprocessor("Date")
    df = date_preprocessor.fit_transform(df)
    df = df.drop(columns=["Date"])

    print(f"Total historical rows: {len(df)}")

    # ─────────────────────────────────────────────
    # Train/test split
    # ─────────────────────────────────────────────
    df_test = df.iloc[-test_size:]
    df_train = df.iloc[:-test_size]

    # ─────────────────────────────────────────────
    # Pipeline — single column, single scaler
    # ─────────────────────────────────────────────
    pipeline = Pipeline([
        ("fillna", SimpleImputer(strategy="mean")),
        ("normalize", MinMaxScaler()),
    ])

    sliding_window = SlidingWindowTransformer(window_size)

    # Scale train data
    train_scaled = pipeline.fit_transform(df_train[[target_col]])
    test_scaled = pipeline.transform(df_test[[target_col]])

    # Get the fitted scaler for inverse transform
    scaler = pipeline.named_steps["normalize"]

    # Create sliding windows
    X_train, y_train = sliding_window.transform(train_scaled)
    X_test, y_test = sliding_window.transform(test_scaled)

    # y is already (n, 1) since we have 1 feature
    y_train = y_train[:, 0:1]
    y_test = y_test[:, 0:1]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # ─────────────────────────────────────────────
    # Train on train set
    # ─────────────────────────────────────────────
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    mlflow.tensorflow.autolog(log_models=False, checkpoint=False)

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
    # Evaluate on test set
    # ─────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)

    print(f"Test MAE:  {mae:.4f}")
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    mlflow.log_metric("test_mae", float(mae))
    mlflow.log_metric("test_mse", float(mse))
    mlflow.log_metric("test_rmse", float(rmse))

    # ─────────────────────────────────────────────
    # Retrain on full dataset
    # ─────────────────────────────────────────────
    full_scaled = pipeline.fit_transform(df[[target_col]])
    scaler_full = pipeline.named_steps["normalize"]

    X_full, y_full = sliding_window.transform(full_scaled)
    y_full = y_full[:, 0:1]

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
    y_full_inv = scaler_full.inverse_transform(y_full.reshape(-1, 1))
    y_pred_full_inv = scaler_full.inverse_transform(y_pred_full)

    mse_full = mean_squared_error(y_full_inv, y_pred_full_inv)
    mae_full = mean_absolute_error(y_full_inv, y_pred_full_inv)
    rmse_full = np.sqrt(mse_full)

    print(f"Full dataset MAE:  {mae_full:.4f}")
    print(f"Full dataset MSE:  {mse_full:.4f}")
    print(f"Full dataset RMSE: {rmse_full:.4f}")

    mlflow.log_metric("full_mae", float(mae_full))
    mlflow.log_metric("full_mse", float(mse_full))
    mlflow.log_metric("full_rmse", float(rmse_full))

    # ─────────────────────────────────────────────
    # Save model and pipeline
    # ─────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    model_path = "models/model_energy_demand_univariate.keras"  # todo: we will also save it in more optimized version
    pipeline_path = "models/pipeline_univariate.pkl"

    model.save(model_path)
    joblib.dump(pipeline, pipeline_path)

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(pipeline_path)

    try:
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model_energy_demand_univariate",
            registered_model_name="energy_demand_univariate_model"
        )
        print("Univariate model registered in MLflow!")
    except Exception as e:
        print(f"WARNING: Could not register model: {e}")

    print("Univariate model and pipeline saved.")
