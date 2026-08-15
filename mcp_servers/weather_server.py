import requests
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server for Weather
mcp = FastMCP("WeatherMCP")

# WMO Weather interpretation codes (WW) mapping
WMO_CODES: Dict[int, str] = {
    0: "Clear sky ☀️",
    1: "Mainly clear 🌤️",
    2: "Partly cloudy ⛅",
    3: "Overcast ☁️",
    45: "Fog 🌫️",
    48: "Depositing rime fog 🌫️",
    51: "Light drizzle 🌧️",
    53: "Moderate drizzle 🌧️",
    55: "Dense drizzle 🌧️",
    56: "Light freezing drizzle 🌧️❄️",
    57: "Dense freezing drizzle 🌧️❄️",
    61: "Slight rain 🌧️",
    63: "Moderate rain 🌧️",
    65: "Heavy rain 🌧️",
    66: "Light freezing rain 🌧️❄️",
    67: "Heavy freezing rain 🌧️❄️",
    71: "Slight snow fall 🌨️",
    73: "Moderate snow fall 🌨️",
    75: "Heavy snow fall 🌨️",
    77: "Snow grains 🌨️",
    80: "Slight rain showers 🌦️",
    81: "Moderate rain showers 🌦️",
    82: "Violent rain showers 🌧️💥",
    85: "Slight snow showers 🌨️",
    86: "Heavy snow showers 🌨️",
    95: "Thunderstorm 🌩️",
    96: "Thunderstorm with slight hail 🌩️🧊",
    99: "Thunderstorm with heavy hail 🌩️🧊",
}


def _geocode_city(city: str) -> Optional[Dict[str, Any]]:
    """
    Helper function to geocode a city name into latitude, longitude, and country name
    using Open-Meteo's free Geocoding API.
    """
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(city)}&count=1&language=en&format=json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results")
            if results and len(results) > 0:
                loc = results[0]
                return {
                    "name": loc.get("name", city),
                    "country": loc.get("country", ""),
                    "admin1": loc.get("admin1", ""),
                    "latitude": loc.get("latitude"),
                    "longitude": loc.get("longitude"),
                    "timezone": loc.get("timezone", "auto"),
                }
    except Exception as e:
        print(f"[WeatherMCP Geocode Error] {e}")
    return None


@mcp.tool()
def get_current_weather(city: str) -> str:
    """
    Get the current weather conditions for a specified destination city.
    
    Args:
        city: Destination city name (e.g. 'Tokyo', 'Paris', 'Delhi', 'New York').
    
    Returns:
        Formatted summary of current weather conditions including temperature, humidity, wind, and conditions.
    """
    geo = _geocode_city(city)
    if not geo:
        return f"Error: Could not locate coordinates for city '{city}'."

    lat = geo["latitude"]
    lon = geo["longitude"]
    location_str = f"{geo['name']}" + (f", {geo['country']}" if geo['country'] else "")

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,"
            f"apparent_temperature,precipitation,weather_code,wind_speed_10m&timezone=auto"
        )
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"Error: Weather API returned status code {response.status_code}."

        data = response.json()
        current = data.get("current", {})
        
        temp = current.get("temperature_2m", "N/A")
        feels_like = current.get("apparent_temperature", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")
        precip = current.get("precipitation", 0.0)
        code = current.get("weather_code", -1)
        condition = WMO_CODES.get(code, "Unknown weather condition")

        return (
            f"🌍 Current Weather Report for {location_str}:\n"
            f"- Condition: {condition}\n"
            f"- Temperature: {temp}°C (Feels like {feels_like}°C)\n"
            f"- Relative Humidity: {humidity}%\n"
            f"- Wind Speed: {wind} km/h\n"
            f"- Precipitation: {precip} mm"
        )
    except Exception as e:
        return f"Error fetching current weather data for '{city}': {str(e)}"


@mcp.tool()
def get_weather_forecast(city: str, days: int = 5) -> str:
    """
    Get a multi-day weather forecast for a destination city.
    
    Args:
        city: Destination city name (e.g. 'Tokyo', 'Mumbai', 'London').
        days: Number of forecast days (1 to 14, default 5).
    
    Returns:
        Formatted daily forecast summary detailing temperatures, weather conditions, precipitation, and wind.
    """
    geo = _geocode_city(city)
    if not geo:
        return f"Error: Could not locate coordinates for city '{city}'."

    lat = geo["latitude"]
    lon = geo["longitude"]
    location_str = f"{geo['name']}" + (f", {geo['country']}" if geo['country'] else "")
    num_days = max(1, min(days, 14))

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=weather_code,temperature_2m_max,"
            f"temperature_2m_min,precipitation_sum,wind_speed_10m_max&forecast_days={num_days}&timezone=auto"
        )
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"Error: Weather API returned status code {response.status_code}."

        data = response.json()
        daily = data.get("daily", {})

        dates = daily.get("time", [])
        codes = daily.get("weather_code", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        wind = daily.get("wind_speed_10m_max", [])

        output = [f"📅 {num_days}-Day Weather Forecast for {location_str}:"]

        for i in range(len(dates)):
            d = dates[i]
            c = WMO_CODES.get(codes[i] if i < len(codes) else -1, "Unknown")
            t_max = temp_max[i] if i < len(temp_max) else "N/A"
            t_min = temp_min[i] if i < len(temp_min) else "N/A"
            p = precip[i] if i < len(precip) else 0
            w = wind[i] if i < len(wind) else "N/A"

            output.append(
                f"• {d}: {c} | Temp: {t_min}°C to {t_max}°C | Rain: {p}mm | Wind Max: {w}km/h"
            )

        return "\n".join(output)
    except Exception as e:
        return f"Error fetching weather forecast for '{city}': {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
