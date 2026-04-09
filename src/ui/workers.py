from PySide6.QtCore import QThread, Signal
from src.api.open_meteo import OpenMeteoClient
from src.ml.pipeline import WeatherModelWrapper
import requests

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
            df_14_days = self.api.get_14_day_dataframe(self.target_date)
            input_tensor = self.model_wrapper.engineer_features(df_14_days)
            predicted_temp = self.model_wrapper.predict(self.model_name, input_tensor)
            actual_temp = self.api.get_actual_temp(self.target_date)
            delta = abs(predicted_temp - actual_temp)
            
            self.finished.emit({
                "predicted": round(predicted_temp, 2),
                "actual": round(actual_temp, 2),
                "delta": round(delta, 2)
            })
        except ValueError as ve:
            self.error.emit(str(ve))
        except requests.exceptions.RequestException:
            self.error.emit("Erro de rede: Falha ao comunicar com a API Open-Meteo.")
        except Exception as e:
            self.error.emit(f"Erro inesperado durante o processamento: {str(e)}")