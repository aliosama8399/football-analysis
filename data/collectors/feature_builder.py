import pandas as pd
import numpy as np
from pathlib import Path

class FeatureBuilder:
    """
    Advanced feature engineering to match the depth of final_dataset.csv.
    Calculates cumulative stats, streaks, and form strings.
    """
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
    def calculate_cumulative_stats(self, df):
        """Calculate cumulative points, goals, etc. for each team through the season"""
        print("Calculating cumulative season stats...")
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Initialize columns
        cols = ['HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'MW', 'HTGD', 'ATGD',
                'DiffPts', 'DiffFormPts', 'DiffFormGoals']
        for col in cols:
            df[col] = 0.0
            
        # Dictionary to track team stats per season
        # Structure: {season: {team: {points: 0, goals_for: 0, ...}}}
        season_stats = {}
        
        for idx, row in df.iterrows():
            season = row.get('Season', 'Unknown') # Need to ensure Season column exists or infer it
            # Fallback if Season distinct not present: assume all one data or rely on Date
            # Better: Reset stats if we see a long gap or infer season from Date
            
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            if season not in season_stats:
                season_stats[season] = {}
            
            if home not in season_stats[season]:
                season_stats[season][home] = {'pts': 0, 'gs': 0, 'gc': 0, 'mw': 0, 'form_pts_5': [], 'form_gf_5': []}
            if away not in season_stats[season]:
                season_stats[season][away] = {'pts': 0, 'gs': 0, 'gc': 0, 'mw': 0, 'form_pts_5': [], 'form_gf_5': []}
                
            # 1. PRE-MATCH STATS (Features for prediction)
            # Assign current cumulative stats to this row
            h_stats = season_stats[season][home]
            a_stats = season_stats[season][away]
            
            df.at[idx, 'HTP'] = h_stats['pts']
            df.at[idx, 'ATP'] = a_stats['pts']
            df.at[idx, 'HTGS'] = h_stats['gs']
            df.at[idx, 'ATGS'] = a_stats['gs']
            df.at[idx, 'HTGC'] = h_stats['gc']
            df.at[idx, 'ATGC'] = a_stats['gc']
            df.at[idx, 'HTGD'] = h_stats['gs'] - h_stats['gc']
            df.at[idx, 'ATGD'] = a_stats['gs'] - a_stats['gc']
            df.at[idx, 'MW'] = h_stats['mw'] + 1 # Upcoming match week
            
            # Form points (last 5)
            ht_form_pts = sum(h_stats['form_pts_5'])
            at_form_pts = sum(a_stats['form_pts_5'])
            df.at[idx, 'HTFormPts'] = ht_form_pts
            df.at[idx, 'ATFormPts'] = at_form_pts
            
            # Differences
            df.at[idx, 'DiffPts'] = h_stats['pts'] - a_stats['pts']
            df.at[idx, 'DiffFormPts'] = ht_form_pts - at_form_pts
            
            # 2. POST-MATCH UPDATES
            h_goals = row['FTHG']
            a_goals = row['FTAG']
            res = row['Result'] # H, D, A
            
            # Update Goals
            season_stats[season][home]['gs'] += h_goals
            season_stats[season][home]['gc'] += a_goals
            season_stats[season][away]['gs'] += a_goals
            season_stats[season][away]['gc'] += h_goals
            
            # Update Points
            h_pts = 3 if res == 'H' else (1 if res == 'D' else 0)
            a_pts = 3 if res == 'A' else (1 if res == 'D' else 0)
            
            season_stats[season][home]['pts'] += h_pts
            season_stats[season][away]['pts'] += a_pts
            
            # Update Match Week
            season_stats[season][home]['mw'] += 1
            season_stats[season][away]['mw'] += 1
            
            # Update Rolling Lists (Form)
            season_stats[season][home]['form_pts_5'].append(h_pts)
            season_stats[season][away]['form_pts_5'].append(a_pts)
            
            # Keep only last 5
            if len(season_stats[season][home]['form_pts_5']) > 5: season_stats[season][home]['form_pts_5'].pop(0)
            if len(season_stats[season][away]['form_pts_5']) > 5: season_stats[season][away]['form_pts_5'].pop(0)

        print("  ✓ Calculated cumulative stats")
        return df

    def calculate_streaks(self, df):
        """Calculate win/loss streaks"""
        print("Calculating streaks...")
        
        df = df.sort_values('Date')
        
        # Init columns
        for col in ['HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
                   'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5']:
            df[col] = 0
            
        season_streaks = {}
        
        for idx, row in df.iterrows():
            season = row.get('Season', 'Unknown')
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            if season not in season_streaks:
                season_streaks[season] = {}
            if home not in season_streaks[season]:
                season_streaks[season][home] = {'win_streak': 0, 'loss_streak': 0, 'history': []}
            if away not in season_streaks[season]:
                season_streaks[season][away] = {'win_streak': 0, 'loss_streak': 0, 'history': []}
                
            # Assign Current Streaks (Pre-match)
            h_str = season_streaks[season][home]
            a_str = season_streaks[season][away]
            
            df.at[idx, 'HTWinStreak3'] = 1 if h_str['win_streak'] >= 3 else 0
            df.at[idx, 'HTWinStreak5'] = 1 if h_str['win_streak'] >= 5 else 0
            df.at[idx, 'HTLossStreak3'] = 1 if h_str['loss_streak'] >= 3 else 0
            df.at[idx, 'HTLossStreak5'] = 1 if h_str['loss_streak'] >= 5 else 0
            
            df.at[idx, 'ATWinStreak3'] = 1 if a_str['win_streak'] >= 3 else 0
            df.at[idx, 'ATWinStreak5'] = 1 if a_str['win_streak'] >= 5 else 0
            df.at[idx, 'ATLossStreak3'] = 1 if a_str['loss_streak'] >= 3 else 0
            df.at[idx, 'ATLossStreak5'] = 1 if a_str['loss_streak'] >= 5 else 0
            
            # Update Streaks (Post-match)
            res = row['Result'] # H, D, A
            
            # Home Updates
            if res == 'H': # Win
                h_str['win_streak'] += 1
                h_str['loss_streak'] = 0
                h_str['history'].append('W')
            elif res == 'A': # Loss
                h_str['win_streak'] = 0
                h_str['loss_streak'] += 1
                h_str['history'].append('L')
            else: # Draw
                h_str['win_streak'] = 0
                h_str['loss_streak'] = 0
                h_str['history'].append('D')
                
            # Away Updates
            if res == 'A': # Win
                a_str['win_streak'] += 1
                a_str['loss_streak'] = 0
                a_str['history'].append('W')
            elif res == 'H': # Loss
                a_str['win_streak'] = 0
                a_str['loss_streak'] += 1
                a_str['history'].append('L')
            else: # Draw
                a_str['win_streak'] = 0
                a_str['loss_streak'] = 0
                a_str['history'].append('D')
                
        print("  ✓ Calculated streaks")
        return df

    def run(self):
        input_path = self.processed_dir / "processed_matches.csv"
        if not input_path.exists():
            print("No processed data found. Run preprocess.py first.")
            return
            
        print(f"Loading {input_path}...")
        df = pd.read_csv(input_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Add Season column if missing (simple logic based on month)
        if 'Season' not in df.columns:
            print("Inferring Season from Date...")
            df['Season'] = df['Date'].apply(lambda x: x.year if x.month > 7 else x.year - 1)
            
        # 1. Cumulative Stats
        df = self.calculate_cumulative_stats(df)
        
        # 2. Streaks
        df = self.calculate_streaks(df)
        
        # Save
        output_path = self.processed_dir / "enhanced_matches.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved enhanced data to {output_path}")
        print(f"  Shape: {df.shape}")
        
        # Also clean up the ML set
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        ml_df = df[numeric_cols].copy()
        # Drop ID-like columns if any, keep targets
        ml_path = self.data_dir / "features" / "enhanced_ml_ready.csv"
        ml_df.to_csv(ml_path, index=False)
        print(f"✓ Saved ML ready data to {ml_path}")

if __name__ == "__main__":
    builder = FeatureBuilder()
    builder.run()
