import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

class WeatherModelWrapper:
    def __init__(self, models_dir="models"):
        # Load scalers
        self.scaler_features = joblib.load(f"{models_dir}/scaler_features.joblib")
        self.scaler_target = joblib.load(f"{models_dir}/scaler_target.joblib")
        
        # Pre-load both models to memory to prevent UI freezing on switch
        self.models = {
            "LSTM": tf.keras.models.load_model(f"{models_dir}/weather_predictor_LSTM.keras"),
            "Transformer": tf.keras.models.load_model(f"{models_dir}/weather_predictor_transformer.keras")
        }
        
        self.feature_order = [
            'temp_max', 'temp_min', 'temp_mean', 'humidity_mean', 'pressure_mean', 
            'precip_total', 'wind_max', 'temp_max_lag_1', 'temp_max_lag_2', 'temp_max_lag_3',
            'temp_max_mean_3d', 'temp_range', 'sin_month', 'cos_month',
            'radiation_sum', 'cloud_cover_mean', 'vpd_max', 'gusts_max', 
            'pressure_tendency', 'precip_accum_7d'
        ]

    def engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        # Calculate derived features
        df['temp_max_lag_1'] = df['temp_max'].shift(1)
        df['temp_max_lag_2'] = df['temp_max'].shift(2)
        df['temp_max_lag_3'] = df['temp_max'].shift(3)
        df['temp_max_mean_3d'] = df['temp_max'].rolling(window=3).mean()
        df['temp_range'] = df['temp_max'] - df['temp_min']
        df['sin_month'] = np.sin(df.index.month * (2. * np.pi / 12))
        df['cos_month'] = np.cos(df.index.month * (2. * np.pi / 12))
        df['pressure_tendency'] = df['pressure_mean'] - df['pressure_mean'].shift(1)
        df['precip_accum_7d'] = df['precip_total'].rolling(window=7).sum()

        # Enforce exact column order
        df = df[self.feature_order]
        
        # Take the last 7 days (this drops the initial 7 days used for building the rolling metrics)
        df_window = df.tail(7)
        
        # Scale and reshape to (1, 7, 20)
        scaled_features = self.scaler_features.transform(df_window.values)
        return scaled_features.reshape(1, 7, 20)

    def predict(self, model_name: str, tensor_input: np.ndarray) -> float:
        model = self.models[model_name]
        scaled_prediction = model.predict(tensor_input, verbose=0)
        
        # Inverse transform the (1, 1) prediction
        celsius_pred = self.scaler_target.inverse_transform(scaled_prediction)
        return float(celsius_pred[0][0])