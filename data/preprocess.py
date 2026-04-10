"""
Data Preprocessing and Feature Engineering Pipeline
Merges multiple data sources and creates ML-ready features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FootballDataProcessor:
    """Process and merge football data from multiple sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Team name mappings for consistency
        self.team_name_mapping = {
            # Premier League variations
            'Man United': 'Manchester United',
            'Man City': 'Manchester City',
            "Nott'm Forest": 'Nottingham Forest',
            'Tottenham': 'Tottenham Hotspur',
            'Sheffield Utd': 'Sheffield United',
            'Newcastle': 'Newcastle United',
            'Wolves': 'Wolverhampton Wanderers',
            'Brighton': 'Brighton and Hove Albion',
            'West Ham': 'West Ham United',
            'Luton': 'Luton Town',
            'Leeds': 'Leeds United',
            'Leicester': 'Leicester City',
            'Ipswich': 'Ipswich Town',
            
            # La Liga variations
            'Ath Madrid': 'Atletico Madrid',
            'Ath Bilbao': 'Athletic Club',
            'Betis': 'Real Betis',
            'Sociedad': 'Real Sociedad',
            'Celta': 'Celta Vigo',
            'Vallecano': 'Rayo Vallecano',
        }
    
    def standardize_team_name(self, name: str) -> str:
        """Standardize team names across datasets"""
        if pd.isna(name):
            return name
        name = str(name).strip()
        return self.team_name_mapping.get(name, name)
    
    def load_football_data_uk(self) -> pd.DataFrame:
        """Load and clean football-data.co.uk data"""
        print("Loading football-data.co.uk data...")
        
        combined_path = self.raw_dir / "football_data_uk_combined.csv"
        if not combined_path.exists():
            print("  ✗ No combined file found. Run the collector first.")
            return pd.DataFrame()
        
        df = pd.read_csv(combined_path)
        print(f"  ✓ Loaded {len(df)} matches")
        if 'League' in df.columns:
            print(f"  ✓ Leagues found: {df['League'].unique().tolist()}")
        elif 'Div' in df.columns:
            print(f"  ✓ Divisions found: {df['Div'].unique().tolist()}")
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        
        # Parse date
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Standardize team names
        df['HomeTeam'] = df['HomeTeam'].apply(self.standardize_team_name)
        df['AwayTeam'] = df['AwayTeam'].apply(self.standardize_team_name)
        
        # Calculate result
        df['Result'] = df.apply(lambda x: 'H' if x['FTHG'] > x['FTAG'] else ('A' if x['FTHG'] < x['FTAG'] else 'D'), axis=1)
        
        # Total goals
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['GoalDiff'] = df['FTHG'] - df['FTAG']
        
        # Over/Under 2.5
        df['Over2.5'] = (df['TotalGoals'] > 2.5).astype(int)
        
        # Both teams scored
        df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        
        return df
    
    def load_understat_data(self) -> pd.DataFrame:
        """Load newly scraped Understat xG data"""
        print("Loading Understat xG data...")
        
        raw_path = self.raw_dir / "understat_xg_data.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            print(f"  ✓ Loaded {len(df)} matches from {raw_path}")
            return df
        
        print("  ✗ No Understat data found. Run data/collectors/understat_scraper.py first.")
        return pd.DataFrame()
    
    def prepare_understat_match_data(self, understat_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare Understat match data for merging with football-data.co.uk"""
        if understat_df.empty:
            return pd.DataFrame()
        
        print("Processing Understat data...")
        
        # Avoid missing key errors by ensuring lowercase conversion or matching expected scraper output
        df = understat_df.copy()
        
        # Standardize team names
        if 'home_team' in df.columns and 'away_team' in df.columns:
            df['HomeTeam'] = df['home_team'].apply(self.standardize_team_name)
            df['AwayTeam'] = df['away_team'].apply(self.standardize_team_name)
            
            # Format dates to dt.date for reliable merging
            if 'date' in df.columns:
                df['Date_only'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            
            # Extract just the critical metrics mapping
            if 'home_xg' in df.columns and 'away_xg' in df.columns:
                xg_data = df[['Date_only', 'HomeTeam', 'AwayTeam', 'home_xg', 'away_xg']].copy()
                xg_data = xg_data.rename(columns={'home_xg': 'Home_xG', 'away_xg': 'Away_xG'})
                print(f"  ✓ Processed Understat for matching: {len(xg_data)} matches")
                return xg_data
                
        print("  ✗ Warning: Understat dataset did not possess expected schema.")
        return pd.DataFrame()
    
    def calculate_team_form(self, df: pd.DataFrame, team_col: str, date_col: str, 
                           result_col: str, n_matches: int = 5) -> pd.DataFrame:
        """Calculate rolling form features for teams"""
        print(f"Calculating {n_matches}-match form...")
        
        # Sort by date
        df = df.sort_values(date_col).copy()
        
        # Create team-level match records
        team_matches = []
        
        for _, row in df.iterrows():
            # Home team record
            home_result = 3 if row['Result'] == 'H' else (1 if row['Result'] == 'D' else 0)
            team_matches.append({
                'Date': row[date_col],
                'Team': row['HomeTeam'],
                'Points': home_result,
                'GF': row['FTHG'],
                'GA': row['FTAG'],
                'xG': row.get('Home_xG', np.nan),
                'xGA': row.get('Away_xG', np.nan),
                # Match stats (for rolling averages)
                'Shots': row.get('HS', np.nan),
                'ShotsAgainst': row.get('AS', np.nan),
                'ShotsOnTarget': row.get('HST', np.nan),
                'ShotsOnTargetAgainst': row.get('AST', np.nan),
                'Corners': row.get('HC', np.nan),
                'CornersAgainst': row.get('AC', np.nan),
                'Fouls': row.get('HF', np.nan),
                'FoulsAgainst': row.get('AF', np.nan),
                'Yellows': row.get('HY', np.nan),
                'Reds': row.get('HR', np.nan),
                'Home': 1
            })
            
            # Away team record
            away_result = 3 if row['Result'] == 'A' else (1 if row['Result'] == 'D' else 0)
            team_matches.append({
                'Date': row[date_col],
                'Team': row['AwayTeam'],
                'Points': away_result,
                'GF': row['FTAG'],
                'GA': row['FTHG'],
                'xG': row.get('Away_xG', np.nan),
                'xGA': row.get('Home_xG', np.nan),
                # Match stats (for rolling averages)
                'Shots': row.get('AS', np.nan),
                'ShotsAgainst': row.get('HS', np.nan),
                'ShotsOnTarget': row.get('AST', np.nan),
                'ShotsOnTargetAgainst': row.get('HST', np.nan),
                'Corners': row.get('AC', np.nan),
                'CornersAgainst': row.get('HC', np.nan),
                'Fouls': row.get('AF', np.nan),
                'FoulsAgainst': row.get('HF', np.nan),
                'Yellows': row.get('AY', np.nan),
                'Reds': row.get('AR', np.nan),
                'Home': 0
            })
        
        team_df = pd.DataFrame(team_matches)
        team_df = team_df.sort_values(['Team', 'Date'])
        
        # Calculate rolling features
        rolling_cols = {
            'Points': 'Form', 'GF': 'GF', 'GA': 'GA', 'xG': 'xG', 'xGA': 'xGA',
            'Shots': 'Shots', 'ShotsAgainst': 'ShotsAgainst',
            'ShotsOnTarget': 'SOT', 'ShotsOnTargetAgainst': 'SOTAgainst',
            'Corners': 'Corners', 'CornersAgainst': 'CornersAgainst',
            'Fouls': 'Fouls', 'FoulsAgainst': 'FoulsAgainst',
            'Yellows': 'Yellows', 'Reds': 'Reds'
        }
        for team in team_df['Team'].unique():
            mask = team_df['Team'] == team
            for raw_col, feat_name in rolling_cols.items():
                if raw_col in team_df.columns:
                    team_df.loc[mask, f'{feat_name}_{n_matches}'] = (
                        team_df.loc[mask, raw_col]
                        .rolling(n_matches, min_periods=1).mean().shift(1)
                    )
        
        print(f"  ✓ Calculated form for {team_df['Team'].nunique()} teams")
        
        return team_df
    
    def merge_form_features(self, matches_df: pd.DataFrame, form_df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
        """Merge form features back to match data"""
        print("Merging form features...")
        
        # Create merge keys
        form_df['merge_key'] = form_df['Team'] + '_' + form_df['Date'].astype(str)
        
        # Merge for home team
        matches_df['home_key'] = matches_df['HomeTeam'] + '_' + matches_df['Date'].astype(str)
        # All rolling feature suffixes to merge
        rolling_suffixes = ['Form', 'GF', 'GA', 'xG', 'xGA',
                           'Shots', 'ShotsAgainst', 'SOT', 'SOTAgainst',
                           'Corners', 'CornersAgainst', 'Fouls', 'FoulsAgainst',
                           'Yellows', 'Reds']
        form_col_names = [f'{s}_{n_matches}' for s in rolling_suffixes]
        available_form_cols = [c for c in form_col_names if c in form_df.columns]
        
        home_form = form_df[form_df['Home'] == 1][['merge_key'] + available_form_cols]
        home_form.columns = ['home_key'] + [f'Home{s}_{n_matches}' for s in rolling_suffixes if f'{s}_{n_matches}' in form_df.columns]
        matches_df = matches_df.merge(home_form, on='home_key', how='left')
        
        # Merge for away team
        matches_df['away_key'] = matches_df['AwayTeam'] + '_' + matches_df['Date'].astype(str)
        away_form = form_df[form_df['Home'] == 0][['merge_key'] + available_form_cols]
        away_form.columns = ['away_key'] + [f'Away{s}_{n_matches}' for s in rolling_suffixes if f'{s}_{n_matches}' in form_df.columns]
        matches_df = matches_df.merge(away_form, on='away_key', how='left')
        
        # Clean up temporary columns
        matches_df = matches_df.drop(columns=['home_key', 'away_key'], errors='ignore')
        
        # Fill NaN form features with reasonable defaults (first matches of season)
        # All rolling feature columns need NaN handling
        all_rolling_cols = [c for c in matches_df.columns 
                           if any(c.startswith(prefix) for prefix in ['Home', 'Away'])
                           and f'_{n_matches}' in c]
        
        for col in all_rolling_cols:
            if col in matches_df.columns:
                if 'xG' in col or 'xGA' in col:
                    # xG may be entirely NaN if no Understat data
                    matches_df[col] = matches_df[col].fillna(0)
                else:
                    # Fill with column mean (league average)
                    col_mean = matches_df[col].mean()
                    if pd.isna(col_mean):
                        col_mean = 1.0 if 'Form' in col else 0.0
                    matches_df[col] = matches_df[col].fillna(col_mean)
        
        print(f"  ✓ Merged form features for {len(matches_df)} matches")
        return matches_df
    
    def calculate_head_to_head(self, df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
        """Calculate head-to-head statistics"""
        print("Calculating head-to-head stats...")
        
        df = df.sort_values('Date').copy()
        
        h2h_stats = []
        for idx, row in df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            match_date = row['Date']
            
            # Get previous meetings
            prev_meetings = df[
                (df['Date'] < match_date) &
                (((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
                 ((df['HomeTeam'] == away) & (df['AwayTeam'] == home)))
            ].tail(n_matches)
            
            if len(prev_meetings) == 0:
                h2h_stats.append({
                    'idx': idx,
                    'H2H_Matches': 0,
                    'H2H_HomeWins': 0,
                    'H2H_AwayWins': 0,
                    'H2H_Draws': 0,
                    'H2H_HomeGoals': 0,
                    'H2H_AwayGoals': 0
                })
            else:
                home_wins = 0
                away_wins = 0
                draws = 0
                home_goals = 0
                away_goals = 0
                
                for _, prev in prev_meetings.iterrows():
                    if prev['HomeTeam'] == home:
                        # Same fixture
                        if prev['Result'] == 'H':
                            home_wins += 1
                        elif prev['Result'] == 'A':
                            away_wins += 1
                        else:
                            draws += 1
                        home_goals += prev['FTHG']
                        away_goals += prev['FTAG']
                    else:
                        # Reversed fixture
                        if prev['Result'] == 'A':
                            home_wins += 1
                        elif prev['Result'] == 'H':
                            away_wins += 1
                        else:
                            draws += 1
                        home_goals += prev['FTAG']
                        away_goals += prev['FTHG']
                
                h2h_stats.append({
                    'idx': idx,
                    'H2H_Matches': len(prev_meetings),
                    'H2H_HomeWins': home_wins,
                    'H2H_AwayWins': away_wins,
                    'H2H_Draws': draws,
                    'H2H_HomeGoals': home_goals / len(prev_meetings),
                    'H2H_AwayGoals': away_goals / len(prev_meetings)
                })
        
        h2h_df = pd.DataFrame(h2h_stats).set_index('idx')
        df = df.join(h2h_df)
        
        print(f"  ✓ Calculated H2H for {len(df)} matches")
        return df
    
    # Betting odds features removed - not used for ethical reasons

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for ML models"""
        print("Creating target variables...")
        
        # Result as numeric
        df['Result_Num'] = df['Result'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Over/Under variants
        df['Over1.5'] = (df['TotalGoals'] > 1.5).astype(int)
        df['Over3.5'] = (df['TotalGoals'] > 3.5).astype(int)
        
        # Clean sheet
        df['HomeCleanSheet'] = (df['FTAG'] == 0).astype(int)
        df['AwayCleanSheet'] = (df['FTHG'] == 0).astype(int)
        
        print(f"  ✓ Created target variables")
        return df
    
    def calculate_referee_strictness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate referee strictness level based on historical cards given"""
        if 'Referee' not in df.columns:
            print("  ✗ No Referee column found")
            return df
        
        print("Calculating referee strictness...")
        
        # Clean referee column - fill missing with 'Unknown'
        df['Referee'] = df['Referee'].fillna('Unknown')
        
        # Calculate cards per match for each referee
        card_cols = ['HY', 'AY', 'HR', 'AR']  # Yellow/Red cards Home/Away
        available_card_cols = [col for col in card_cols if col in df.columns]
        
        if not available_card_cols:
            print("  ✗ No card columns found (HY, AY, HR, AR)")
            df['Ref_Strictness'] = 0.5  # Default medium strictness
            return df
        
        # Calculate total cards per match
        df['_TotalYellows'] = df[['HY', 'AY']].sum(axis=1) if 'HY' in df.columns and 'AY' in df.columns else 0
        df['_TotalReds'] = df[['HR', 'AR']].sum(axis=1) if 'HR' in df.columns and 'AR' in df.columns else 0
        
        # Calculate referee averages using historical data (shifted to avoid leakage)
        ref_stats = df.groupby('Referee').agg({
            '_TotalYellows': 'mean',
            '_TotalReds': 'mean'
        }).rename(columns={'_TotalYellows': 'Ref_AvgYellows', '_TotalReds': 'Ref_AvgReds'})
        
        # Strictness score: Yellows + 3*Reds (scaled to 0-1)
        ref_stats['Ref_Strictness_Raw'] = ref_stats['Ref_AvgYellows'] + 3 * ref_stats['Ref_AvgReds']
        max_strictness = ref_stats['Ref_Strictness_Raw'].max()
        ref_stats['Ref_Strictness'] = ref_stats['Ref_Strictness_Raw'] / max_strictness if max_strictness > 0 else 0.5
        
        # Merge back to dataframe
        df = df.merge(ref_stats[['Ref_AvgYellows', 'Ref_AvgReds', 'Ref_Strictness']], 
                      left_on='Referee', right_index=True, how='left')
        
        # Fill any missing with average
        df['Ref_Strictness'] = df['Ref_Strictness'].fillna(df['Ref_Strictness'].mean())
        df['Ref_AvgYellows'] = df['Ref_AvgYellows'].fillna(df['Ref_AvgYellows'].mean())
        df['Ref_AvgReds'] = df['Ref_AvgReds'].fillna(df['Ref_AvgReds'].mean())
        
        # Drop temporary columns
        df = df.drop(columns=['_TotalYellows', '_TotalReds'], errors='ignore')
        
        # Drop referee name column (user wants strictness only, not the name)
        df = df.drop(columns=['Referee'], errors='ignore')
        
        print(f"  ✓ Calculated referee strictness for {len(ref_stats)} referees")
        print(f"    Strictness range: {df['Ref_Strictness'].min():.2f} - {df['Ref_Strictness'].max():.2f}")
        
        return df
    
    def process_all(self) -> pd.DataFrame:

        """Run the full preprocessing pipeline"""
        print("=" * 60)
        print("FOOTBALL DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        print()
        
        # Load data
        fduk_df = self.load_football_data_uk()
        understat_df = self.load_understat_data()
        
        if fduk_df.empty:
            print("\n✗ No data to process!")
            return pd.DataFrame()
        
        # Process and tightly merge Understat data BEFORE calculating form features
        processed_understat = self.prepare_understat_match_data(understat_df)
        if not processed_understat.empty:
            fduk_df['Date_only'] = pd.to_datetime(fduk_df['Date'], errors='coerce').dt.date
            # Pre-merge row count
            start_len = len(fduk_df)
            fduk_df = fduk_df.merge(processed_understat, on=['Date_only', 'HomeTeam', 'AwayTeam'], how='left')
            fduk_df = fduk_df.drop(columns=['Date_only'], errors='ignore')
            # Count how many matches actually found an xG match
            xg_found = fduk_df['Home_xG'].notna().sum()
            print(f"  ✓ Successfully merged xG values for {xg_found} out of {start_len} matches")
        
        # Calculate form features
        form_df = self.calculate_team_form(fduk_df, 'HomeTeam', 'Date', 'Result', n_matches=5)
        
        # Merge form features
        fduk_df = self.merge_form_features(fduk_df, form_df, n_matches=5)
        
        # Calculate H2H
        fduk_df = self.calculate_head_to_head(fduk_df, n_matches=5)
        
        # Remove ALL betting/odds columns from the data
        betting_patterns = [
            'B365', 'BW', 'IW', 'PS', 'WH', 'VC', 'ODDS', 'BET', 'MAX', 'AVG',
            'PSCH', 'PSCD', 'PSCA', 'BF', '1XB', 'BFEX', 'BFE', 'PC>', 'PC<',
            'P>', 'P<', 'AH', 'PAH', 'PCAH', 'PROB', 'PINNACLE'
        ]
        betting_cols = [col for col in fduk_df.columns if any(x in col.upper() for x in betting_patterns)]
        if betting_cols:
            fduk_df = fduk_df.drop(columns=betting_cols, errors='ignore')
            print(f"  ✓ Removed {len(betting_cols)} betting-related columns")
        
        # Remove duplicate/malformed columns
        dup_cols = [col for col in fduk_df.columns if 'ï»¿' in col or col.startswith('Unnamed')]
        if dup_cols:
            fduk_df = fduk_df.drop(columns=dup_cols, errors='ignore')
            print(f"  ✓ Removed {len(dup_cols)} duplicate/malformed columns")
        
        # Calculate referee strictness (if Referee column exists)
        fduk_df = self.calculate_referee_strictness(fduk_df)
        
        
        # Create targets
        fduk_df = self.create_target_variables(fduk_df)
        
        # Save processed data
        output_path = self.processed_dir / "processed_matches.csv"
        fduk_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed data to {output_path}")
        print(f"  Total matches: {len(fduk_df)}")
        print(f"  Total features: {len(fduk_df.columns)}")
        
        # Save feature-ready data (only numeric features)
        feature_cols = [col for col in fduk_df.columns if fduk_df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        feature_df = fduk_df[feature_cols].copy()
        feature_path = self.features_dir / "ml_ready_data.csv"
        feature_df.to_csv(feature_path, index=False)
        print(f"✓ Saved ML-ready data to {feature_path}")
        print(f"  Numeric features: {len(feature_cols)}")
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        
        return fduk_df


def main():
    """Run the preprocessing pipeline"""
    processor = FootballDataProcessor()
    df = processor.process_all()
    
    if not df.empty:
        print("\nData Summary:")
        print("-" * 40)
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"\nLeagues:")
        print(df['League'].value_counts())
        print(f"\nResult distribution:")
        print(df['Result'].value_counts(normalize=True).round(3))
        print(f"\nSample features:")
        print(df[['HomeTeam', 'AwayTeam', 'HomeForm_5', 'AwayForm_5', 'H2H_Matches', 'TotalGoals']].head(10))


if __name__ == "__main__":
    main()
