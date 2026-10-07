"""Cities, dates and Open-Meteo variable names used across the project."""
from datetime import date

CITIES = {
    "surat": (21.17, 72.83),
    "ahmedabad": (23.03, 72.58),
}

# Diwali (Lakshmi Puja) dates inside the data range.
DIWALI = {
    2022: date(2022, 10, 24),
    2023: date(2023, 11, 12),
    2024: date(2024, 10, 31),
    2025: date(2025, 10, 20),
}

# Open-Meteo's global air quality model (CAMS) has data from here on.
START = date(2022, 8, 1)

AIR_VARS = ["pm2_5", "pm10", "nitrogen_dioxide"]
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
]


def display_name(city: str) -> str:
    return city.capitalize()
