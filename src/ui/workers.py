from PySide6.QtCore import QThread, Signal
from src.api.open_meteo import OpenMeteoClient
from src.ml.pipeline import WeatherModelWrapper
import pandas as pd

class PredictionWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, target_date: str, model_name: str, model_wrapper: WeatherModelWrapper):
        super().__init__()
        self.target_date = target_date
        self.model_name = model_name
        self.model_wrapper = model_wrapper
        self.api = OpenMeteoClient()

    def run(self):
        try:
            # 1. Fetch Data
            df_14_days = self.api.get_14_day_dataframe(self.target_date)
            
            # 2. Build Tensor
            input_tensor = self.model_wrapper.engineer_features(df_14_days)
            
            # 3. Predict
            predicted_temp = self.model_wrapper.predict(self.model_name, input_tensor)
            
            # 4. Get Ground Truth
            actual_temp = self.api.get_actual_temp(self.target_date)
            
            # 5. Math
            delta = abs(predicted_temp - actual_temp)
            
            self.finished.emit({
                "predicted": round(predicted_temp, 2),
                "actual": round(actual_temp, 2),
                "delta": round(delta, 2)
            })
        except Exception as e:
            self.error.emit(f"Error processing prediction: {str(e)}")