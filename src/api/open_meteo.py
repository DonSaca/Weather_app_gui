import requests
import requests_cache
import pandas as pd
from datetime import datetime, timedelta

requests_cache.install_cache('meteo_cache', expire_after=3600)

class OpenMeteoClient:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.lat = 38.7167  # Lisbon
        self.lon = -9.1333  # Lisbon
        self.timezone = "Europe/Lisbon"

    def get_14_day_dataframe(self, target_date: str) -> pd.DataFrame:
        target = datetime.strptime(target_date, "%Y-%m-%d")
        # Fetch 14 days prior to target date to build 7-day accumulators and lags
        start_date = (target - timedelta(days=14)).strftime("%Y-%m-%d")
        end_date = (target - timedelta(days=1)).strftime("%Y-%m-%d")

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": self.timezone,
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "surface_pressure", 
                "precipitation", "wind_speed_10m", "shortwave_radiation", 
                "cloud_cover", "vapor_pressure_deficit", "wind_gusts_10m"
            ]
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        hourly_data = response.json()['hourly']
        df = pd.DataFrame(hourly_data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Resample hourly to daily aggregates
        daily_df = df.resample('D').agg(
            temp_max=('temperature_2m', 'max'),
            temp_min=('temperature_2m', 'min'),
            temp_mean=('temperature_2m', 'mean'),
            humidity_mean=('relative_humidity_2m', 'mean'),
            pressure_mean=('surface_pressure', 'mean'),
            precip_total=('precipitation', 'sum'),
            wind_max=('wind_speed_10m', 'max'),
            radiation_sum=('shortwave_radiation', 'sum'),
            cloud_cover_mean=('cloud_cover', 'mean'),
            vpd_max=('vapor_pressure_deficit', 'max'),
            gusts_max=('wind_gusts_10m', 'max')
        )
        return daily_df

    def get_actual_temp(self, target_date: str) -> float:
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": target_date,
            "end_date": target_date,
            "daily": "temperature_2m_max",
            "timezone": self.timezone
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()['daily']['temperature_2m_max'][0]