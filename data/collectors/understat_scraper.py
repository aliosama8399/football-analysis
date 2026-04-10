import soccerdata as sd
from pathlib import Path

def main():
    print("Fetching Understat Match Data for Top 5 Leagues...")
    leagues = ['ENG-Premier League', 'ESP-La Liga', 'ITA-Serie A', 'GER-Bundesliga', 'FRA-Ligue 1']
    seasons = ['2022-2023', '2023-2024', '2024-2025']
    
    u = sd.Understat(leagues, seasons=seasons)
    df = u.read_team_match_stats()
    
    # Save the raw data
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "understat_xg_data.csv"
    
    df.reset_index().to_csv(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")

if __name__ == "__main__":
    main()
