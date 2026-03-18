"""
Main Data Collection Script
Runs all data collectors and merges the data
"""

import sys
from pathlib import Path

# Add collectors to path
sys.path.insert(0, str(Path(__file__).parent / "collectors"))

import pandas as pd
from datetime import datetime


def run_all_collectors():
    """Run all data collectors"""
    print("=" * 70)
    print("FOOTBALL DATA COLLECTION PIPELINE")
    print("Leagues: Premier League, La Liga")
    print("Seasons: 2022-23, 2023-24, 2024-25")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: Download from football-data.co.uk
    print("\n" + "=" * 70)
    print("STEP 1: Downloading from football-data.co.uk")
    print("=" * 70)
    
    from collectors.football_data_uk import download_all_data
    fduk_df = download_all_data()
    
    # Step 2: Collect Understat xG data
    print("\n" + "=" * 70)
    print("STEP 2: Collecting Understat xG data")
    print("=" * 70)
    
    from collectors.understat_collector import run_collector
    understat_matches, understat_teams = run_collector()
    
    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nFiles saved to data/raw/:")
    print("  - football_data_uk_combined.csv")
    print("  - understat_matches.csv")
    print("  - understat_team_stats.csv")
    
    return fduk_df, understat_matches, understat_teams


def collect_football_data_uk():
    """Collect only football-data.co.uk data"""
    from collectors.football_data_uk import download_all_data
    return download_all_data()


def collect_understat_data():
    """Collect only Understat data"""
    from collectors.understat_collector import run_collector
    return run_collector()


if __name__ == "__main__":
    run_all_collectors()
