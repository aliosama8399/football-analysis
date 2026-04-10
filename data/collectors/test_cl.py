"""
Champions League Data Collector (ScraperFC FBRef)
==================================================
Scrapes the last 10 seasons of UEFA Champions League match data from FBRef
via ScraperFC, then runs the full feature-engineering pipeline to produce
a file with the same 65 columns as ml_ready_data.csv.

Usage:
    python data/collectors/champions_league.py

Output:
    data/processed/cl_processed_matches.csv   (full features, text cols)
    data/features/cl_ml_ready_data.csv         (numeric only, same schema as ml_ready_data.csv)
"""

import sys, os, time, warnings, re
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # football/
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
FEAT_DIR = BASE_DIR / "data" / "features"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# Last 10 CL seasons (FBRef string format → internal season code)
SEASONS = {
    # '2015-2016': '1516',
    '2016-2017': # '1617',
    '2017-2018': # '1718',
    '2018-2019': # '1819',
    '2019-2020': # '1920',
    '2020-2021': # '2021',
    '2021-2022': # '2122',
    '2022-2023': # '2223',
    '2023-2024': '2324',
    '2024-2025': # '2425',
}

LEAGUE = 'UEFA Champions League'


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Scrape the Scores-and-Fixtures page (one per season)
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_season_schedule(fbref, season_str: str) -> pd.DataFrame:
    """
    Scrape the FBRef Scores & Fixtures page for a given CL season.
    Returns a DataFrame with one row per match, containing:
      Date, HomeTeam, AwayTeam, home score, away score, xG, Referee, etc.
    """
    from bs4 import BeautifulSoup

    valid = fbref.get_valid_seasons(LEAGUE)
    if season_str not in valid:
        print(f"  ✗ Season {season_str} not available on FBRef")
        return pd.DataFrame()

    season_url = valid[season_str]
    # Build the fixtures URL
    parts = season_url.split("/")
    parts.insert(-1, "schedule")
    fixtures_url = "/".join(parts).replace("Stats", "Scores-and-Fixtures")

    print(f"  Fetching: {fixtures_url}")
    soup = fbref._get_soup(fixtures_url)

    # Parse the main schedule table
    tables = pd.read_html(StringIO(str(soup)))

    # The schedule table usually has columns like:
    # Wk, Day, Date, Time, Home, xG, Score, xG, Away, Attendance, Venue, Referee, Report, Notes
    # Find the correct table by looking for 'Score' column
    schedule_df = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any('Score' in c for c in cols):
            schedule_df = t
            break

    if schedule_df is None:
        print(f"  ✗ Could not find schedule table for {season_str}")
        return pd.DataFrame()

    # Flatten multi-level columns if present
    if isinstance(schedule_df.columns, pd.MultiIndex):
        schedule_df.columns = [
            '_'.join(str(c) for c in col).strip('_') for col in schedule_df.columns
        ]

    print(f"  ✓ Found schedule with {len(schedule_df)} rows, columns: {list(schedule_df.columns)}")
    return schedule_df


def parse_schedule_to_matches(schedule_df: pd.DataFrame, season_code: str) -> pd.DataFrame:
    """
    Convert a raw FBRef schedule table into our standard match format.
    Handles varying column names across seasons.
    """
    if schedule_df.empty:
        return pd.DataFrame()

    cols = list(schedule_df.columns)
    col_lower = {c: c.lower() for c in cols}

    # Helper to find a column by substring
    def find_col(*patterns):
        for c in cols:
            cl = c.lower()
            for p in patterns:
                if p in cl:
                    return c
        return None

    date_col = find_col('date')
    home_col = find_col('home')
    away_col = find_col('away')
    score_col = find_col('score')
    referee_col = find_col('referee')

    # xG columns: usually two columns both containing 'xg'
    xg_cols = [c for c in cols if 'xg' in c.lower()]

    if not all([date_col, home_col, away_col, score_col]):
        print(f"  ✗ Missing essential columns. Found: {cols}")
        return pd.DataFrame()

    matches = []
    for _, row in schedule_df.iterrows():
        try:
            score_str = str(row[score_col])
            # Score format: "2–1" or "2-1" or "2–1 (a.e.t.)" etc.
            score_str = score_str.split('(')[0].strip()  # remove extra info
            score_parts = re.split(r'[–\-]', score_str)
            if len(score_parts) != 2:
                continue
            fthg = int(score_parts[0].strip())
            ftag = int(score_parts[1].strip())
        except (ValueError, TypeError):
            continue

        home_team = str(row[home_col]).strip()
        away_team = str(row[away_col]).strip()
        if home_team == 'nan' or away_team == 'nan':
            continue

        # Clean FBRef country codes (e.g. 'eng Manchester City' or 'Barcelona es')
        home_team = re.sub(r'^[a-z]{2,3}\s+|\s+[a-z]{2,3}$', '', home_team)
        away_team = re.sub(r'^[a-z]{2,3}\s+|\s+[a-z]{2,3}$', '', away_team)

        # Determine result
        if fthg > ftag:
            ftr = 'H'
        elif ftag > fthg:
            ftr = 'A'
        else:
            ftr = 'D'

        match = {
            'Date': row[date_col],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTHG': fthg,
            'FTAG': ftag,
            'FTR': ftr,
            'Season': season_code,
            'League': 'Champions_League',
            'Div': 'CL',
        }

        # xG (often two columns: home xG, away xG)
        if len(xg_cols) >= 2:
            try:
                match['Home_xG'] = float(row[xg_cols[0]])
            except (ValueError, TypeError):
                match['Home_xG'] = np.nan
            try:
                match['Away_xG'] = float(row[xg_cols[1]])
            except (ValueError, TypeError):
                match['Away_xG'] = np.nan
        else:
            match['Home_xG'] = np.nan
            match['Away_xG'] = np.nan

        # Referee
        if referee_col:
            ref = str(row[referee_col])
            match['Referee'] = ref if ref != 'nan' else 'Unknown'
        else:
            match['Referee'] = 'Unknown'

        matches.append(match)

    df = pd.DataFrame(matches)
    print(f"  ✓ Parsed {len(df)} valid matches from schedule")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Scrape squad-level misc stats for fouls, cards, corners
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_season_squad_stats(fbref, season_str: str) -> dict:
    """
    Scrape the squad standard, misc, and shooting stats for a CL season.
    Returns dict: {'standard': df, 'misc': df, 'shooting': df}
    """
    stats = {}
    for cat in ['standard', 'misc', 'shooting']:
        try:
            result = fbref.scrape_stats(season_str, LEAGUE, cat)
            stats[cat] = result
            time.sleep(3)
        except Exception as e:
            print(f"  ✗ Error scraping {cat} stats for {season_str}: {e}")
            stats[cat] = None
    return stats


def build_team_averages(stats: dict) -> pd.DataFrame:
    """
    Build per-team averages for shots, fouls, corners, cards, xG.
    Returns a DataFrame indexed by cleaned Team name.
    """
    team_data = {}

    def clean_team_name(name):
        return re.sub(r'^[a-z]{2,3}\s+|\s+[a-z]{2,3}$', '', str(name)).strip()

    # Dictionary mapping what we want: (category, possible_col_names)
    target_stats = {
        'xG': ('standard', ['Expected_xG', 'xG']),
        'Shots': ('shooting', ['Standard_Sh', 'Sh']),
        'SOT': ('shooting', ['Standard_SoT', 'SoT']),
        'Fouls': ('misc', ['Performance_Fls', 'Fls']),
        'Corners': ('misc', ['Performance_Crs', 'Crs']),
        'Yellows': ('misc', ['Performance_CrdY', 'CrdY']),
        'Reds': ('misc', ['Performance_CrdR', 'CrdR']),
        'Matches': ('standard', ['Playing Time_MP', 'MP'])
    }

    # Extract tables from ScraperFC dictionary structure
    for cat_name, cat_data in stats.items():
        if not cat_data: continue
        # Usually squad stats are in 'Squad Standard Stats' or similar
        squad_df = None
        if isinstance(cat_data, dict):
            for k, v in cat_data.items():
                if 'squad' in k.lower():
                    squad_df = v
                    break
            # Fallback to first df if 'squad' not in keys
            if squad_df is None and len(cat_data) > 0:
                squad_df = list(cat_data.values())[0]
        else:
            squad_df = cat_data

        if squad_df is None or squad_df.empty: continue

        # Flatten columns if multi-index
        if isinstance(squad_df.columns, pd.MultiIndex):
            squad_df.columns = ['_'.join(str(c) for c in col if 'Unnamed' not in str(c)).strip('_') for col in squad_df.columns]

        # Process each team
        for _, row in squad_df.iterrows():
            # ScraperFC usually has a 'Squad' column
            team_col = next((c for c in squad_df.columns if 'squad' in c.lower() or 'team' in c.lower()), None)
            if not team_col: continue

            team = clean_team_name(row[team_col])
            if team not in team_data:
                team_data[team] = {}

            # Extract our target stats
            for stat_key, (target_cat, possible_cols) in target_stats.items():
                if cat_name == target_cat:
                    for pcol in possible_cols:
                        if pcol in squad_df.columns:
                            try:
                                team_data[team][stat_key] = float(row[pcol])
                            except (ValueError, TypeError):
                                team_data[team][stat_key] = 0.0
                            break

    df = pd.DataFrame.from_dict(team_data, orient='index')
    
    # Calculate per-match averages
    if 'Matches' in df.columns and len(df) > 0:
        # Avoid division by zero
        mp = df['Matches'].replace(0, 1)
        for c in ['xG', 'Shots', 'SOT', 'Fouls', 'Corners', 'Yellows', 'Reds']:
            if c in df.columns:
                df[c] = df[c] / mp
            else:
                df[c] = 0.0
    return df


def inject_team_stats(matches_df: pd.DataFrame, team_avg_df: pd.DataFrame) -> pd.DataFrame:
    """Inject season-level team averages into the match-level dataframe."""
    new_df = matches_df.copy()
    
    for idx, row in new_df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Inject per-match averages for Home team
        if home in team_avg_df.index:
            h_stats = team_avg_df.loc[home]
            new_df.at[idx, 'Home_xG'] = float(h_stats.get('xG', 0))
            new_df.at[idx, 'HS'] = float(h_stats.get('Shots', 0))
            new_df.at[idx, 'HST'] = float(h_stats.get('SOT', 0))
            new_df.at[idx, 'HF'] = float(h_stats.get('Fouls', 0))
            new_df.at[idx, 'HC'] = float(h_stats.get('Corners', 0))
            new_df.at[idx, 'HY'] = float(h_stats.get('Yellows', 0))
            new_df.at[idx, 'HR'] = float(h_stats.get('Reds', 0))
            
        # Inject per-match averages for Away team
        if away in team_avg_df.index:
            a_stats = team_avg_df.loc[away]
            new_df.at[idx, 'Away_xG'] = float(a_stats.get('xG', 0))
            new_df.at[idx, 'AS'] = float(a_stats.get('Shots', 0))
            new_df.at[idx, 'AST'] = float(a_stats.get('SOT', 0))
            new_df.at[idx, 'AF'] = float(a_stats.get('Fouls', 0))
            new_df.at[idx, 'AC'] = float(a_stats.get('Corners', 0))
            new_df.at[idx, 'AY'] = float(a_stats.get('Yellows', 0))
            new_df.at[idx, 'AR'] = float(a_stats.get('Reds', 0))
            
    return new_df


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Feature Engineering (reuse preprocess.py logic)
# ═══════════════════════════════════════════════════════════════════════════════

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns from goals."""
    df['Result'] = df['FTR']
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['GoalDiff'] = df['FTHG'] - df['FTAG']
    df['Over2.5'] = (df['TotalGoals'] > 2.5).astype(int)
    df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    # Fill missing match stat columns with NaN (they'll be filled later)
    for col in ['HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
                'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'Time']:
        if col not in df.columns:
            df[col] = np.nan
    return df


def calculate_rolling_form(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """Calculate rolling 5-match form for all teams (same logic as preprocess.py)."""
    print(f"  Calculating {n_matches}-match rolling form...")
    df = df.sort_values('Date').copy()

    team_matches = []
    for _, row in df.iterrows():
        # Home team record
        home_pts = 3 if row['Result'] == 'H' else (1 if row['Result'] == 'D' else 0)
        team_matches.append({
            'Date': row['Date'], 'Team': row['HomeTeam'], 'Points': home_pts,
            'GF': row['FTHG'], 'GA': row['FTAG'],
            'xG': row.get('Home_xG', np.nan), 'xGA': row.get('Away_xG', np.nan),
            'Shots': row.get('HS', np.nan), 'ShotsAgainst': row.get('AS', np.nan),
            'ShotsOnTarget': row.get('HST', np.nan), 'ShotsOnTargetAgainst': row.get('AST', np.nan),
            'Corners': row.get('HC', np.nan), 'CornersAgainst': row.get('AC', np.nan),
            'Fouls': row.get('HF', np.nan), 'FoulsAgainst': row.get('AF', np.nan),
            'Yellows': row.get('HY', np.nan), 'Reds': row.get('HR', np.nan),
            'Home': 1,
        })
        # Away team record
        away_pts = 3 if row['Result'] == 'A' else (1 if row['Result'] == 'D' else 0)
        team_matches.append({
            'Date': row['Date'], 'Team': row['AwayTeam'], 'Points': away_pts,
            'GF': row['FTAG'], 'GA': row['FTHG'],
            'xG': row.get('Away_xG', np.nan), 'xGA': row.get('Home_xG', np.nan),
            'Shots': row.get('AS', np.nan), 'ShotsAgainst': row.get('HS', np.nan),
            'ShotsOnTarget': row.get('AST', np.nan), 'ShotsOnTargetAgainst': row.get('HST', np.nan),
            'Corners': row.get('AC', np.nan), 'CornersAgainst': row.get('HC', np.nan),
            'Fouls': row.get('AF', np.nan), 'FoulsAgainst': row.get('HF', np.nan),
            'Yellows': row.get('AY', np.nan), 'Reds': row.get('AR', np.nan),
            'Home': 0,
        })

    team_df = pd.DataFrame(team_matches)
    team_df = team_df.sort_values(['Team', 'Date'])

    rolling_map = {
        'Points': 'Form', 'GF': 'GF', 'GA': 'GA', 'xG': 'xG', 'xGA': 'xGA',
        'Shots': 'Shots', 'ShotsAgainst': 'ShotsAgainst',
        'ShotsOnTarget': 'SOT', 'ShotsOnTargetAgainst': 'SOTAgainst',
        'Corners': 'Corners', 'CornersAgainst': 'CornersAgainst',
        'Fouls': 'Fouls', 'FoulsAgainst': 'FoulsAgainst',
        'Yellows': 'Yellows', 'Reds': 'Reds',
    }
    for team in team_df['Team'].unique():
        mask = team_df['Team'] == team
        for raw_col, feat in rolling_map.items():
            if raw_col in team_df.columns:
                team_df.loc[mask, f'{feat}_{n_matches}'] = (
                    team_df.loc[mask, raw_col]
                    .rolling(n_matches, min_periods=1).mean().shift(1)
                )

    print(f"    ✓ Form for {team_df['Team'].nunique()} teams")
    return team_df


def merge_form(df: pd.DataFrame, form_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Merge rolling form back to match rows."""
    form_df['merge_key'] = form_df['Team'] + '_' + form_df['Date'].astype(str)
    suffixes = ['Form', 'GF', 'GA', 'xG', 'xGA', 'Shots', 'ShotsAgainst',
                'SOT', 'SOTAgainst', 'Corners', 'CornersAgainst',
                'Fouls', 'FoulsAgainst', 'Yellows', 'Reds']
    form_cols = [f'{s}_{n}' for s in suffixes if f'{s}_{n}' in form_df.columns]

    # Home
    df['home_key'] = df['HomeTeam'] + '_' + df['Date'].astype(str)
    home = form_df[form_df['Home'] == 1][['merge_key'] + form_cols].copy()
    home.columns = ['home_key'] + [f'Home{s}_{n}' for s in suffixes if f'{s}_{n}' in form_df.columns]
    df = df.merge(home, on='home_key', how='left')

    # Away
    df['away_key'] = df['AwayTeam'] + '_' + df['Date'].astype(str)
    away = form_df[form_df['Home'] == 0][['merge_key'] + form_cols].copy()
    away.columns = ['away_key'] + [f'Away{s}_{n}' for s in suffixes if f'{s}_{n}' in form_df.columns]
    df = df.merge(away, on='away_key', how='left')
    df.drop(columns=['home_key', 'away_key'], errors='ignore', inplace=True)

    # Fill NaN
    for col in df.columns:
        if f'_{n}' in col and any(col.startswith(p) for p in ['Home', 'Away']):
            if 'xG' in col:
                df[col] = df[col].fillna(0)
            else:
                m = df[col].mean()
                df[col] = df[col].fillna(m if not pd.isna(m) else 0)

    return df


def calculate_h2h(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Head-to-head stats (same logic as preprocess.py)."""
    print("  Calculating H2H stats...")
    df = df.sort_values('Date').copy()
    h2h = []
    for idx, row in df.iterrows():
        home, away, date = row['HomeTeam'], row['AwayTeam'], row['Date']
        prev = df[
            (df['Date'] < date) &
            (((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
             ((df['HomeTeam'] == away) & (df['AwayTeam'] == home)))
        ].tail(n)
        if len(prev) == 0:
            h2h.append({'idx': idx, 'H2H_Matches': 0, 'H2H_HomeWins': 0,
                         'H2H_AwayWins': 0, 'H2H_Draws': 0,
                         'H2H_HomeGoals': 0, 'H2H_AwayGoals': 0})
        else:
            hw, aw, dr, hg, ag = 0, 0, 0, 0, 0
            for _, p in prev.iterrows():
                if p['HomeTeam'] == home:
                    if p['Result'] == 'H': hw += 1
                    elif p['Result'] == 'A': aw += 1
                    else: dr += 1
                    hg += p['FTHG']; ag += p['FTAG']
                else:
                    if p['Result'] == 'A': hw += 1
                    elif p['Result'] == 'H': aw += 1
                    else: dr += 1
                    hg += p['FTAG']; ag += p['FTHG']
            h2h.append({'idx': idx, 'H2H_Matches': len(prev), 'H2H_HomeWins': hw,
                         'H2H_AwayWins': aw, 'H2H_Draws': dr,
                         'H2H_HomeGoals': hg / len(prev),
                         'H2H_AwayGoals': ag / len(prev)})
    h2h_df = pd.DataFrame(h2h).set_index('idx')
    df = df.join(h2h_df)
    print(f"    ✓ H2H for {len(df)} matches")
    return df


def calculate_referee_strictness(df: pd.DataFrame) -> pd.DataFrame:
    """Referee strictness features (same logic as preprocess.py)."""
    print("  Calculating referee strictness...")
    if 'Referee' not in df.columns:
        df['Ref_AvgYellows'] = 0
        df['Ref_AvgReds'] = 0
        df['Ref_Strictness'] = 0.5
        return df

    df['Referee'] = df['Referee'].fillna('Unknown')

    # Use HY/AY/HR/AR if available, else fill 0
    for c in ['HY', 'AY', 'HR', 'AR']:
        if c not in df.columns:
            df[c] = 0

    df['_TotalY'] = df[['HY', 'AY']].sum(axis=1)
    df['_TotalR'] = df[['HR', 'AR']].sum(axis=1)

    ref = df.groupby('Referee').agg({'_TotalY': 'mean', '_TotalR': 'mean'})
    ref.columns = ['Ref_AvgYellows', 'Ref_AvgReds']
    ref['_raw'] = ref['Ref_AvgYellows'] + 3 * ref['Ref_AvgReds']
    mx = ref['_raw'].max()
    ref['Ref_Strictness'] = ref['_raw'] / mx if mx > 0 else 0.5
    ref = ref.drop(columns=['_raw'])

    df = df.merge(ref, left_on='Referee', right_index=True, how='left')
    df['Ref_Strictness'] = df['Ref_Strictness'].fillna(df['Ref_Strictness'].mean())
    df['Ref_AvgYellows'] = df['Ref_AvgYellows'].fillna(df['Ref_AvgYellows'].mean())
    df['Ref_AvgReds'] = df['Ref_AvgReds'].fillna(df['Ref_AvgReds'].mean())
    df.drop(columns=['_TotalY', '_TotalR'], errors='ignore', inplace=True)
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables."""
    df['Result_Num'] = df['Result'].map({'H': 0, 'D': 1, 'A': 2})
    df['Over1.5'] = (df['TotalGoals'] > 1.5).astype(int)
    df['Over3.5'] = (df['TotalGoals'] > 3.5).astype(int)
    df['HomeCleanSheet'] = (df['FTAG'] == 0).astype(int)
    df['AwayCleanSheet'] = (df['FTHG'] == 0).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import ScraperFC

    print("=" * 70)
    print("  CHAMPIONS LEAGUE DATA COLLECTION — ScraperFC / FBRef")
    print(f"  Seasons: {list(SEASONS.keys())}")
    print("=" * 70)

    fbref = ScraperFC.FBref(wait_time=8)  # 8s between requests to avoid ban

    all_matches = []

    for season_str, season_code in SEASONS.items():
        print(f"\n{'─' * 60}")
        print(f"  Season: {season_str} (code: {season_code})")
        print(f"{'─' * 60}")

        # Step 1: Scrape match schedule
        try:
            schedule_df = scrape_season_schedule(fbref, season_str)
        except Exception as e:
            print(f"  ✗ Error scraping season {season_str}: {e}")
            continue

        if schedule_df.empty:
            print(f"  ✗ No schedule data for {season_str}")
            continue

        # Step 2: Parse into our format
        matches = parse_schedule_to_matches(schedule_df, season_code)

        if not matches.empty:
            # Step 1.5: Scrape squad stats and calculate per-match averages
            squad_stats = scrape_season_squad_stats(fbref, season_str)
            team_avgs = build_team_averages(squad_stats)
            
            # Inject the granular match stats into the schedule matches
            if not team_avgs.empty:
                matches = inject_team_stats(matches, team_avgs)
                print(f"  ✓ Injected team stats for {len(matches)} matches")

            # Save raw per-season CSV
            raw_path = RAW_DIR / f"cl_raw_{season_code}.csv"
            matches.to_csv(raw_path, index=False)
            print(f"  ✓ Saved raw season data to {raw_path.name}")
            all_matches.append(matches)

        # Rate limit between seasons
        time.sleep(5)

    if not all_matches:
        print("\n✗ No Champions League data collected!")
        return

    # Combine all seasons
    df = pd.concat(all_matches, ignore_index=True)
    print(f"\n{'=' * 60}")
    print(f"  COMBINED: {len(df)} matches across {len(all_matches)} seasons")
    print(f"{'=' * 60}")

    # Save combined raw
    df.to_csv(RAW_DIR / "champions_league_combined.csv", index=False)

    # ── Feature Engineering ──
    print("\n  Running Feature Engineering Pipeline...")

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Basic features
    df = add_basic_features(df)

    # Rolling 5-match form
    form_df = calculate_rolling_form(df, n_matches=5)
    df = merge_form(df, form_df, n=5)

    # H2H stats
    df = calculate_h2h(df, n=5)

    # Referee strictness
    df = calculate_referee_strictness(df)

    # Target variables
    df = create_targets(df)

    # Drop non-numeric / non-essential text columns for the ml_ready file
    df.drop(columns=['Referee'], errors='ignore', inplace=True)

    # Save processed (with text columns)
    df.to_csv(PROC_DIR / "cl_processed_matches.csv", index=False)
    print(f"\n  ✓ Saved: data/processed/cl_processed_matches.csv  ({len(df)} rows)")

    # Save ML-ready (numeric only) — same schema as ml_ready_data.csv
    target_cols = [
        'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
        'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'Season', 'TotalGoals', 'GoalDiff',
        'Over2.5', 'BTTS',
        'HomeForm_5', 'HomeGF_5', 'HomeGA_5', 'HomexG_5', 'HomexGA_5',
        'HomeShots_5', 'HomeShotsAgainst_5', 'HomeSOT_5', 'HomeSOTAgainst_5',
        'HomeCorners_5', 'HomeCornersAgainst_5', 'HomeFouls_5', 'HomeFoulsAgainst_5',
        'HomeYellows_5', 'HomeReds_5',
        'AwayForm_5', 'AwayGF_5', 'AwayGA_5', 'AwayxG_5', 'AwayxGA_5',
        'AwayShots_5', 'AwayShotsAgainst_5', 'AwaySOT_5', 'AwaySOTAgainst_5',
        'AwayCorners_5', 'AwayCornersAgainst_5', 'AwayFouls_5', 'AwayFoulsAgainst_5',
        'AwayYellows_5', 'AwayReds_5',
        'H2H_Matches', 'H2H_HomeWins', 'H2H_AwayWins', 'H2H_Draws',
        'H2H_HomeGoals', 'H2H_AwayGoals',
        'Ref_AvgYellows', 'Ref_AvgReds', 'Ref_Strictness',
        'Result_Num', 'Over1.5', 'Over3.5', 'HomeCleanSheet', 'AwayCleanSheet'
    ]

    # Ensure all target columns exist (fill missing with NaN → 0)
    for c in target_cols:
        if c not in df.columns:
            df[c] = 0

    ml_df = df[target_cols].copy()
    # Convert Season codes to numeric
    ml_df['Season'] = pd.to_numeric(ml_df['Season'], errors='coerce').fillna(0).astype(int)

    ml_df.to_csv(FEAT_DIR / "cl_ml_ready_data.csv", index=False)
    print(f"  ✓ Saved: data/features/cl_ml_ready_data.csv  ({len(ml_df)} rows, {len(target_cols)} cols)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"  Total matches: {len(df)}")
    print(f"  Unique teams: {df['HomeTeam'].nunique()}")
    print(f"  Columns: {len(target_cols)} (matches ml_ready_data.csv)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
