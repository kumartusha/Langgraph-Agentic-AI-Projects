import requests
import random
from typing import Dict
from src.config.settings import settings

class WeatherService:
    @staticmethod
    def get_weather(city: str, simulate: bool = False) -> Dict:
        """
        Fetches live weather from OpenWeatherMap, or returns simulated data for testing.
        """
        if simulate:
            # Return simulated high-severity weather to force the agent down the critical path
            return {
                "weather": "violent storm with heavy rain",
                "wind_speed": random.uniform(30.0, 50.0), # Hurricane force
                "cloud_cover": 100,
                "sea_level": 1005,
                "temperature": random.uniform(5.0, 15.0),
                "humidity": 95,
                "pressure": 980
            }
            
        BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
        request_url = f"{BASE_URL}?appid={settings.WEATHER_API_KEY}&q={city}"
        
        try:
            response = requests.get(request_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "weather": data.get('weather', [{}])[0].get("description", "N/A"),
                "wind_speed": data.get("wind", {}).get("speed", "N/A"),
                "cloud_cover": data.get("clouds", {}).get("all", "N/A"),
                "sea_level": data.get("main", {}).get("sea_level", "N/A"),
                "temperature": round(data.get("main", {}).get("temp", 273.15) - 273.15, 1),
                "humidity": data.get("main", {}).get("humidity", "N/A"),
                "pressure": data.get("main", {}).get("pressure", "N/A")
            }
        except Exception as e:
            raise RuntimeError(f"Failed to fetch weather data for {city}: {str(e)}")
