import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

def get_asset_path(relative_path):
    """Resolve correct path for PyInstaller or local execution."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class WeatherModelWrapper:
    def __init__(self, models_dir="models"):
        # Resolve paths
        scaler_feat_path = get_asset_path(f"{models_dir}/scaler_features.joblib")
        scaler_tgt_path = get_asset_path(f"{models_dir}/scaler_target.joblib")
        lstm_path = get_asset_path(f"{models_dir}/weather_predictor_LSTM.keras")
        transformer_path = get_asset_path(f"{models_dir}/weather_predictor_transformer.keras")

        # Validation: Alert immediately if assets are missing
        for path in [scaler_feat_path, scaler_tgt_path, lstm_path, transformer_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Ficheiro de modelo em falta: {path}")

        # Load scalers
        self.scaler_features = joblib.load(scaler_feat_path)
        self.scaler_target = joblib.load(scaler_tgt_path)
        
        # Pre-load models
        self.models = {
            "LSTM": tf.keras.models.load_model(lstm_path),
            "Transformer": tf.keras.models.load_model(transformer_path)
        }
        
        self.feature_order = [
            'temp_max', 'temp_min', 'temp_mean', 'humidity_mean', 'pressure_mean', 
            'precip_total', 'wind_max', 'temp_max_lag_1', 'temp_max_lag_2', 'temp_max_lag_3',
            'temp_max_mean_3d', 'temp_range', 'sin_month', 'cos_month',
            'radiation_sum', 'cloud_cover_mean', 'vpd_max', 'gusts_max', 
            'pressure_tendency', 'precip_accum_7d'
        ]

    def engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        df['temp_max_lag_1'] = df['temp_max'].shift(1)
        df['temp_max_lag_2'] = df['temp_max'].shift(2)
        df['temp_max_lag_3'] = df['temp_max'].shift(3)
        df['temp_max_mean_3d'] = df['temp_max'].rolling(window=3).mean()
        df['temp_range'] = df['temp_max'] - df['temp_min']
        df['sin_month'] = np.sin(df.index.month * (2. * np.pi / 12))
        df['cos_month'] = np.cos(df.index.month * (2. * np.pi / 12))
        df['pressure_tendency'] = df['pressure_mean'] - df['pressure_mean'].shift(1)
        df['precip_accum_7d'] = df['precip_total'].rolling(window=7).sum()

        df = df[self.feature_order]
        df_window = df.tail(7)
        
        scaled_features = self.scaler_features.transform(df_window.values)
        return scaled_features.reshape(1, 7, 20)

    def predict(self, model_name: str, tensor_input: np.ndarray) -> float:
        model = self.models[model_name]
        scaled_prediction = model.predict(tensor_input, verbose=0)
        celsius_pred = self.scaler_target.inverse_transform(scaled_prediction)
        return float(celsius_pred[0][0])