"""
Champions League Data Collector
Downloads match data from openfootball GitHub repository and formats it to match football-data.co.uk format.
repo: https://github.com/openfootball/champions-league
"""

import pandas as pd
import requests
from io import StringIO
from pathlib import Path
import time

# Base URL for openfootball champions league data
BASE_URL = "https://raw.githubusercontent.com/footballcsv/europe-champions-league/master"

# Seasons to download (e.g., 2022-23, 2023-24)
# Note: openfootball structure uses directory names like '2022-23'
SEASONS = {
    '2223': '2022-23',
    '2324': '2023-24'
}

def download_cl_season(season_code: str, season_dir: str) -> pd.DataFrame:
    """
    Download Champions League data for a specific season.
    
    Args:
        season_code: Our internal code (e.g., '2223')
        season_dir: Directory name in github repo (e.g., '2022-23')
    """
    # The file name is usually 'cl.csv' or 'champions-league.csv' inside the season directory
    # We might need to try a few variations or check the repo structure
    # Based on openfootball, it often has `cl.csv` within the season folder.
    
    url = f"{BASE_URL}/{season_dir}/cl.csv"
    print(f"Downloading: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        # Try alternative filename if first fails
        if response.status_code == 404:
             url = f"{BASE_URL}/{season_dir}/champions-league.csv"
             print(f"Retrying with: {url}")
             response = requests.get(url, timeout=30)
             
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        print(f"  ✓ Downloaded {len(df)} matches")
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading CL {season_code}: {e}")
        return pd.DataFrame()

def format_cl_data(df: pd.DataFrame, season_code: str) -> pd.DataFrame:
    """
    Format openfootball data to match football-data.co.uk structure
    
    OpenFootball columns (typical):
    Round, Date, Team 1, FT, Team 2
    
    Target columns:
    Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Season
    """
    if df.empty:
        return df
        
    processed_data = []
    
    for _, row in df.iterrows():
        try:
            # Parse score '2-1', '1-1', etc.
            score = row.get('FT', '')
            if not isinstance(score, str) or '-' not in score:
                continue
                
            goals = score.split('-')
            fthg = int(goals[0])
            ftag = int(goals[1])
            
            # Determine result
            if fthg > ftag:
                ftr = 'H'
            elif ftag > fthg:
                ftr = 'A'
            else:
                ftr = 'D'
            
            processed_data.append({
                'Div': 'AC', # Arbitrary code for All Champions League
                'Date': row.get('Date'),
                'HomeTeam': row.get('Team 1'),
                'AwayTeam': row.get('Team 2'),
                'FTHG': fthg,
                'FTAG': ftag,
                'FTR': ftr,
                'Season': season_code,
                'League': 'Champions_League'
            })
        except Exception:
            continue
            
    return pd.DataFrame(processed_data)

def download_champions_league(output_dir: str = "data/raw") -> pd.DataFrame:
    """Download and merge all requested CL seasons"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_seasons = []
    
    for code, directory in SEASONS.items():
        df = download_cl_season(code, directory)
        if not df.empty:
            formatted_df = format_cl_data(df, code)
            if not formatted_df.empty:
                filename = f"Champions_League_{code}.csv"
                formatted_df.to_csv(output_path / filename, index=False)
                print(f"  Saved formatted data to {filename}")
                all_seasons.append(formatted_df)
        
        time.sleep(1)
        
    if all_seasons:
        return pd.concat(all_seasons, ignore_index=True)
    
    return pd.DataFrame()

if __name__ == "__main__":
    download_champions_league()
