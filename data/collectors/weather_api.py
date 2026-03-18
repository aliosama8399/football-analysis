"""
Weather Data Collector
Gets historical weather data for football matches using Open-Meteo API (free, no key required)
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time

# Stadium coordinates for Premier League and La Liga teams
STADIUM_COORDS = {
    # Premier League
    'Arsenal': {'lat': 51.5549, 'lon': -0.1084, 'stadium': 'Emirates Stadium'},
    'Aston Villa': {'lat': 52.5092, 'lon': -1.8846, 'stadium': 'Villa Park'},
    'Bournemouth': {'lat': 50.7352, 'lon': -1.8384, 'stadium': 'Vitality Stadium'},
    'Brentford': {'lat': 51.4907, 'lon': -0.2886, 'stadium': 'Gtech Community Stadium'},
    'Brighton': {'lat': 50.8619, 'lon': -0.0837, 'stadium': 'Amex Stadium'},
    'Burnley': {'lat': 53.7890, 'lon': -2.2304, 'stadium': 'Turf Moor'},
    'Chelsea': {'lat': 51.4817, 'lon': -0.1909, 'stadium': 'Stamford Bridge'},
    'Crystal Palace': {'lat': 51.3984, 'lon': -0.0855, 'stadium': 'Selhurst Park'},
    'Everton': {'lat': 53.4389, 'lon': -2.9664, 'stadium': 'Goodison Park'},
    'Fulham': {'lat': 51.4749, 'lon': -0.2217, 'stadium': 'Craven Cottage'},
    'Ipswich': {'lat': 52.0545, 'lon': 1.1449, 'stadium': 'Portman Road'},
    'Ipswich Town': {'lat': 52.0545, 'lon': 1.1449, 'stadium': 'Portman Road'},
    'Leeds': {'lat': 53.7780, 'lon': -1.5720, 'stadium': 'Elland Road'},
    'Leeds United': {'lat': 53.7780, 'lon': -1.5720, 'stadium': 'Elland Road'},
    'Leicester': {'lat': 52.6204, 'lon': -1.1422, 'stadium': 'King Power Stadium'},
    'Leicester City': {'lat': 52.6204, 'lon': -1.1422, 'stadium': 'King Power Stadium'},
    'Liverpool': {'lat': 53.4308, 'lon': -2.9608, 'stadium': 'Anfield'},
    'Luton': {'lat': 51.8842, 'lon': -0.4317, 'stadium': 'Kenilworth Road'},
    'Luton Town': {'lat': 51.8842, 'lon': -0.4317, 'stadium': 'Kenilworth Road'},
    'Man City': {'lat': 53.4831, 'lon': -2.2004, 'stadium': 'Etihad Stadium'},
    'Man United': {'lat': 53.4631, 'lon': -2.2913, 'stadium': 'Old Trafford'},
    'Manchester City': {'lat': 53.4831, 'lon': -2.2004, 'stadium': 'Etihad Stadium'},
    'Manchester Utd': {'lat': 53.4631, 'lon': -2.2913, 'stadium': 'Old Trafford'},
    'Newcastle': {'lat': 54.9756, 'lon': -1.6217, 'stadium': 'St James Park'},
    'Newcastle United': {'lat': 54.9756, 'lon': -1.6217, 'stadium': 'St James Park'},
    'Nott\'m Forest': {'lat': 52.9400, 'lon': -1.1328, 'stadium': 'City Ground'},
    'Nottingham Forest': {'lat': 52.9400, 'lon': -1.1328, 'stadium': 'City Ground'},
    'Sheffield Utd': {'lat': 53.3703, 'lon': -1.4709, 'stadium': 'Bramall Lane'},
    'Sheffield United': {'lat': 53.3703, 'lon': -1.4709, 'stadium': 'Bramall Lane'},
    'Southampton': {'lat': 50.9058, 'lon': -1.3910, 'stadium': 'St Marys Stadium'},
    'Tottenham': {'lat': 51.6042, 'lon': -0.0662, 'stadium': 'Tottenham Hotspur Stadium'},
    'West Ham': {'lat': 51.5387, 'lon': -0.0166, 'stadium': 'London Stadium'},
    'Wolves': {'lat': 52.5903, 'lon': -2.1306, 'stadium': 'Molineux'},
    'Wolverhampton': {'lat': 52.5903, 'lon': -2.1306, 'stadium': 'Molineux'},
    
    # La Liga
    'Alaves': {'lat': 42.8371, 'lon': -2.6880, 'stadium': 'Mendizorrotza'},
    'Almeria': {'lat': 36.8401, 'lon': -2.4362, 'stadium': 'Power Horse Stadium'},
    'Athletic Club': {'lat': 43.2641, 'lon': -2.9493, 'stadium': 'San Mames'},
    'Ath Bilbao': {'lat': 43.2641, 'lon': -2.9493, 'stadium': 'San Mames'},
    'Ath Madrid': {'lat': 40.4362, 'lon': -3.5994, 'stadium': 'Wanda Metropolitano'},
    'Atletico Madrid': {'lat': 40.4362, 'lon': -3.5994, 'stadium': 'Wanda Metropolitano'},
    'Barcelona': {'lat': 41.3809, 'lon': 2.1228, 'stadium': 'Camp Nou'},
    'Betis': {'lat': 37.3566, 'lon': -5.9817, 'stadium': 'Benito Villamarin'},
    'Real Betis': {'lat': 37.3566, 'lon': -5.9817, 'stadium': 'Benito Villamarin'},
    'Cadiz': {'lat': 36.5027, 'lon': -6.2728, 'stadium': 'Nuevo Mirandilla'},
    'Celta': {'lat': 42.2119, 'lon': -8.7394, 'stadium': 'Balaidos'},
    'Celta Vigo': {'lat': 42.2119, 'lon': -8.7394, 'stadium': 'Balaidos'},
    'Elche': {'lat': 38.2671, 'lon': -0.6632, 'stadium': 'Martinez Valero'},
    'Espanyol': {'lat': 41.3479, 'lon': 2.0757, 'stadium': 'RCDE Stadium'},
    'Getafe': {'lat': 40.3256, 'lon': -3.7148, 'stadium': 'Coliseum Alfonso Perez'},
    'Girona': {'lat': 41.9619, 'lon': 2.8287, 'stadium': 'Estadi Montilivi'},
    'Granada': {'lat': 37.1530, 'lon': -3.5956, 'stadium': 'Nuevo Los Carmenes'},
    'Las Palmas': {'lat': 28.1004, 'lon': -15.4567, 'stadium': 'Gran Canaria'},
    'Leganes': {'lat': 40.3575, 'lon': -3.7595, 'stadium': 'Butarque'},
    'Levante': {'lat': 39.4954, 'lon': -0.3640, 'stadium': 'Ciutat de Valencia'},
    'Mallorca': {'lat': 39.5901, 'lon': 2.6312, 'stadium': 'Visit Mallorca Estadi'},
    'Osasuna': {'lat': 42.7967, 'lon': -1.6368, 'stadium': 'El Sadar'},
    'Rayo Vallecano': {'lat': 40.3920, 'lon': -3.6587, 'stadium': 'Campo de Vallecas'},
    'Real Madrid': {'lat': 40.4530, 'lon': -3.6883, 'stadium': 'Santiago Bernabeu'},
    'Real Sociedad': {'lat': 43.3013, 'lon': -1.9736, 'stadium': 'Reale Arena'},
    'Sevilla': {'lat': 37.3840, 'lon': -5.9705, 'stadium': 'Ramon Sanchez Pizjuan'},
    'Valencia': {'lat': 39.4745, 'lon': -0.3583, 'stadium': 'Mestalla'},
    'Valladolid': {'lat': 41.6444, 'lon': -4.7612, 'stadium': 'Jose Zorrilla'},
    'Villarreal': {'lat': 39.9441, 'lon': -0.1037, 'stadium': 'Estadio de la Ceramica'},
}


def get_historical_weather(lat: float, lon: float, date: str, hour: int = 15) -> dict:
    """
    Get historical weather from Open-Meteo API (free, no key required)
    
    Args:
        lat: Latitude
        lon: Longitude
        date: Date string in YYYY-MM-DD format
        hour: Hour of day to get weather for (default 15:00)
    
    Returns:
        Dictionary with weather data
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,precipitation,rain,wind_speed_10m,relative_humidity_2m",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'hourly' in data and data['hourly']['time']:
            return {
                'temperature': data['hourly']['temperature_2m'][hour],
                'precipitation': data['hourly']['precipitation'][hour],
                'rain': data['hourly']['rain'][hour],
                'wind_speed': data['hourly']['wind_speed_10m'][hour],
                'humidity': data['hourly']['relative_humidity_2m'][hour]
            }
    except Exception as e:
        print(f"Weather API error for {date}: {e}")
    
    return {
        'temperature': None,
        'precipitation': None,
        'rain': None,
        'wind_speed': None,
        'humidity': None
    }


def add_weather_to_matches(matches_df: pd.DataFrame, date_col: str = 'Date', 
                           home_team_col: str = 'HomeTeam') -> pd.DataFrame:
    """
    Add weather data to a matches DataFrame
    
    Args:
        matches_df: DataFrame with match data
        date_col: Name of the date column
        home_team_col: Name of the home team column
    
    Returns:
        DataFrame with weather columns added
    """
    print(f"Adding weather data to {len(matches_df)} matches...")
    
    weather_data = []
    
    for idx, row in matches_df.iterrows():
        home_team = row[home_team_col]
        match_date = row[date_col]
        
        # Parse date
        if isinstance(match_date, str):
            try:
                # Try different date formats
                for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d/%m/%y']:
                    try:
                        dt = datetime.strptime(match_date, fmt)
                        date_str = dt.strftime('%Y-%m-%d')
                        break
                    except:
                        continue
                else:
                    date_str = None
            except:
                date_str = None
        else:
            date_str = None
        
        # Get stadium coordinates
        coords = STADIUM_COORDS.get(home_team)
        
        if coords and date_str:
            weather = get_historical_weather(coords['lat'], coords['lon'], date_str)
        else:
            weather = {
                'temperature': None,
                'precipitation': None,
                'rain': None,
                'wind_speed': None,
                'humidity': None
            }
        
        weather_data.append(weather)
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(matches_df)} matches")
        
        # Rate limiting - Open-Meteo allows 10,000 requests/day
        time.sleep(0.2)
    
    # Add weather columns to DataFrame
    weather_df = pd.DataFrame(weather_data)
    result_df = pd.concat([matches_df.reset_index(drop=True), weather_df], axis=1)
    
    print(f"✓ Added weather data to {len(result_df)} matches")
    return result_df


def get_weather_for_date_location(team: str, date: str) -> dict:
    """
    Get weather for a specific team and date
    
    Args:
        team: Team name
        date: Date string (YYYY-MM-DD)
    
    Returns:
        Weather dictionary
    """
    coords = STADIUM_COORDS.get(team)
    if coords:
        return get_historical_weather(coords['lat'], coords['lon'], date)
    return {}


if __name__ == "__main__":
    # Test the weather API
    print("Testing Open-Meteo Weather API...")
    
    # Test for a specific match
    test_team = "Liverpool"
    test_date = "2024-01-01"
    
    print(f"\nWeather for {test_team} on {test_date}:")
    weather = get_weather_for_date_location(test_team, test_date)
    
    for key, value in weather.items():
        print(f"  {key}: {value}")
    
    print("\nStadium database contains:")
    print(f"  {len([t for t in STADIUM_COORDS if 'lat' in STADIUM_COORDS[t]])} stadiums")
