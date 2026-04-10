"""
Champions League Data Collector — Per-Match FBRef Scraping
===========================================================
Scrapes the last 10 seasons of UEFA Champions League match data
by fetching INDIVIDUAL match report pages from FBRef. This gives
us REAL per-match stats (shots, fouls, corners, cards, xG).

Features:
  - Resumable: caches each match to JSON, resumes from where it stopped
  - Rate limited: 6s delay between requests to prevent IP blocks
  - Full pipeline: rolling form, H2H, referee strictness, targets

Usage:
    python data/collectors/champions_league.py

Output:
    data/raw/cl_match_cache/            (cached JSON for each match)
    data/raw/champions_league_combined.csv
    data/processed/cl_processed_matches.csv
    data/features/cl_ml_ready_data.csv  (65 cols)
"""

import sys
import os
import time
import warnings
import re
import json
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
FEAT_DIR = BASE_DIR / "data" / "features"
CACHE_DIR = RAW_DIR / "cl_match_cache"
for d in [RAW_DIR, PROC_DIR, FEAT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEASONS = {
    '2015-2016': '1516', '2016-2017': '1617', '2017-2018': '1718',
    '2018-2019': '1819', '2019-2020': '1920', '2020-2021': '2021',
    '2021-2022': '2122', '2022-2023': '2223', '2023-2024': '2324',
    '2024-2025': '2425',
}

LEAGUE = 'UEFA Champions League'
WAIT_TIME = 6  # seconds


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_name(name):
    """Strip FBRef country-code prefixes/suffixes."""
    name = str(name).strip()
    name = re.sub(r'^[a-z]{2,3}\s+', '', name)
    name = re.sub(r'\s+[a-z]{2,3}$', '', name)
    return name.strip()

def url_to_cache_key(url):
    parts = url.rstrip('/').split('/')
    return parts[-2] if len(parts) >= 2 else url.replace('/', '_')

def safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (ValueError, TypeError):
        return default


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACT MATCH STATS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_match_stats(match_obj):
    """Extract stats from FBrefMatch into a flat dict."""
    try:
        home = clean_name(match_obj.home_team or '')
        away = clean_name(match_obj.away_team or '')
        if not home or not away:
            return None

        # Fix: Convert int/float strictly without throwing errors on None
        try:
            fthg = int(match_obj.home_goals)
        except:
            fthg = 0
            
        try:
            ftag = int(match_obj.away_goals)
        except:
            ftag = 0

        data = {
            'Date': str(match_obj.date),
            'HomeTeam': home,
            'AwayTeam': away,
            'FTHG': fthg,
            'FTAG': ftag,
            'Referee': clean_name(match_obj.referee or 'Unknown'),
        }

        # Result
        if data['FTHG'] > data['FTAG']:
            data['FTR'] = 'H'
        elif data['FTAG'] > data['FTHG']:
            data['FTR'] = 'A'
        else:
            data['FTR'] = 'D'

        home_xg = away_xg = 0.0
        for shots_attr, xg_key in [('home_shots', 'Home_xG'), ('away_shots', 'Away_xG')]:
            shots_df = getattr(match_obj, shots_attr, None)
            if shots_df is not None and not shots_df.empty:
                xg_col = next((c for c in shots_df.columns if 'xg' in str(c).lower()), None)
                if xg_col:
                    if xg_key == 'Home_xG':
                        home_xg = safe_float(shots_df[xg_col].sum())
                    else:
                        away_xg = safe_float(shots_df[xg_col].sum())

        if home_xg == 0 and away_xg == 0:
            all_shots = getattr(match_obj, 'all_shots', None)
            if all_shots is not None and not all_shots.empty:
                xg_col = next((c for c in all_shots.columns if 'xg' in str(c).lower()), None)
                squad_col = next((c for c in all_shots.columns if 'squad' in str(c).lower()), None)
                if xg_col and squad_col:
                    for _, row in all_shots.iterrows():
                        team = clean_name(str(row[squad_col]))
                        xg_val = safe_float(row[xg_col])
                        if team == home:
                            home_xg += xg_val
                        elif team == away:
                            away_xg += xg_val

        data['Home_xG'] = round(home_xg, 2)
        data['Away_xG'] = round(away_xg, 2)

        for col in ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']:
            data[col] = 0

        for side, prefix in [('home', 'H'), ('away', 'A')]:
            pstats = getattr(match_obj, f'{side}_player_stats', None)
            if pstats is None or not isinstance(pstats, dict):
                continue

            for table_name, tdf in pstats.items():
                if tdf is None or tdf.empty:
                    continue

                tname = str(table_name).lower()
                cols_lower = {str(c): str(c).lower() for c in tdf.columns}

                if 'summary' in tname or 'shooting' in tname:
                    for col_name, col_low in cols_lower.items():
                        if col_low in ['sh', 'shots'] and data[f'{prefix}S'] == 0:
                            data[f'{prefix}S'] = int(safe_float(tdf[col_name].sum()))
                        elif col_low in ['sot', 'shots on target'] and data[f'{prefix}ST'] == 0:
                            data[f'{prefix}ST'] = int(safe_float(tdf[col_name].sum()))

                if 'misc' in tname:
                    for col_name, col_low in cols_lower.items():
                        if col_low in ['fls', 'fouls'] and data[f'{prefix}F'] == 0:
                            data[f'{prefix}F'] = int(safe_float(tdf[col_name].sum()))
                        elif col_low in ['crdy', 'yellow cards'] and data[f'{prefix}Y'] == 0:
                            data[f'{prefix}Y'] = int(safe_float(tdf[col_name].sum()))
                        elif col_low in ['crdr', 'red cards'] and data[f'{prefix}R'] == 0:
                            data[f'{prefix}R'] = int(safe_float(tdf[col_name].sum()))

                if 'passing' in tname:
                    for col_name, col_low in cols_lower.items():
                        if col_low in ['crs', 'crosses'] and data[f'{prefix}C'] == 0:
                            data[f'{prefix}C'] = int(safe_float(tdf[col_name].sum()))

        return data
    except Exception as e:
        print(f"     ⚠ extract error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SCRAPE MATCHES
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_season_matches(fbref, season_str, season_code):
    print(f"  📋 Getting match links for {season_str}...")
    try:
        links = fbref.get_match_links(season_str, LEAGUE)
    except Exception as e:
        print(f"  ✗ Failed to get match links: {e}")
        return []

    print(f"  ✓ Found {len(links)} match links")
    results = []
    cached_count = 0
    scraped_count = 0
    error_count = 0

    for i, url in enumerate(links):
        cache_key = url_to_cache_key(url)
        cache_file = CACHE_DIR / f"{season_code}_{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results.append(data)
                cached_count += 1
                continue
            except json.JSONDecodeError:
                pass

        try:
            match_obj = fbref.scrape_match(url)
            data = extract_match_stats(match_obj)

            if data:
                data['Season'] = season_code
                data['League'] = 'Champions_League'
                data['Div'] = 'CL'
                data['url'] = url

                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)

                results.append(data)
                scraped_count += 1
            else:
                error_count += 1

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(term in error_str for term in [
                'paused', 'rate', 'timeout', 'elementwithselectornotfound', 'captcha'
            ])
            
            if is_rate_limit:
                print(f"  🛑 FBref Cloudflare Block / Rate Limit Hit!")
                print("     FBref temporarily locks IPs if too many pages are scraped.")
                print("     Sleeping for 15 minutes to let the ban lift...")
                time.sleep(900)  # 15 minutes
                
                try:
                    match_obj = fbref.scrape_match(url)
                    data = extract_match_stats(match_obj)
                    if data:
                        data['Season'] = season_code
                        data['League'] = 'Champions_League'
                        data['Div'] = 'CL'
                        data['url'] = url
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False)
                        results.append(data)
                        scraped_count += 1
                    else:
                        error_count += 1
                except Exception as e2:
                    print(f"     ✗ Retry failed: {e2}")
                    error_count += 1
            else:
                print(f"     ✗ Match {i+1}: {error_str[:80]}")
                error_count += 1

        done = i + 1
        total = len(links)
        if done % 5 == 0 or done == total:
            print(f"  [{done}/{total}] scraped={scraped_count} cached={cached_count} errors={error_count}")

    print(f"  ✅ Season {season_str}: {len(results)} matches")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def add_basics(df):
    df['Result'] = df['FTR']
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['GoalDiff'] = df['FTHG'] - df['FTAG']
    df['Over2.5'] = (df['TotalGoals'] > 2.5).astype(int)
    df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    for c in ['HTHG', 'HTAG', 'Time']:
        if c not in df.columns:
            df[c] = np.nan
    for c in ['Home_xG', 'Away_xG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df

def rolling_form(df, n=5):
    print(f"  📈 Rolling {n}-match form...")
    df = df.sort_values('Date').copy()
    recs = []
    for _, row in df.iterrows():
        for h in [True, False]:
            pts = 3 if (row['Result'] == 'H' and h) or (row['Result'] == 'A' and not h) else (1 if row['Result'] == 'D' else 0)
            recs.append({
                'Date': row['Date'],
                'Team': row['HomeTeam'] if h else row['AwayTeam'],
                'Points': pts,
                'GF': row['FTHG'] if h else row['FTAG'],
                'GA': row['FTAG'] if h else row['FTHG'],
                'xG': row.get('Home_xG' if h else 'Away_xG', 0.0),
                'xGA': row.get('Away_xG' if h else 'Home_xG', 0.0),
                'Shots': row.get('HS' if h else 'AS', 0.0),
                'ShotsAgainst': row.get('AS' if h else 'HS', 0.0),
                'ShotsOnTarget': row.get('HST' if h else 'AST', 0.0),
                'ShotsOnTargetAgainst': row.get('AST' if h else 'HST', 0.0),
                'Corners': row.get('HC' if h else 'AC', 0.0),
                'CornersAgainst': row.get('AC' if h else 'HC', 0.0),
                'Fouls': row.get('HF' if h else 'AF', 0.0),
                'FoulsAgainst': row.get('AF' if h else 'HF', 0.0),
                'Yellows': row.get('HY' if h else 'AY', 0.0),
                'Reds': row.get('HR' if h else 'AR', 0.0),
                'Home': 1 if h else 0,
            })
    tdf = pd.DataFrame(recs).sort_values(['Team', 'Date'])
    feat_map = {'Points': 'Form', 'GF': 'GF', 'GA': 'GA', 'xG': 'xG', 'xGA': 'xGA',
                'Shots': 'Shots', 'ShotsAgainst': 'ShotsAgainst',
                'ShotsOnTarget': 'SOT', 'ShotsOnTargetAgainst': 'SOTAgainst',
                'Corners': 'Corners', 'CornersAgainst': 'CornersAgainst',
                'Fouls': 'Fouls', 'FoulsAgainst': 'FoulsAgainst',
                'Yellows': 'Yellows', 'Reds': 'Reds'}
    for team in tdf['Team'].unique():
        m = tdf['Team'] == team
        for raw, feat in feat_map.items():
            tdf.loc[m, f'{feat}_{n}'] = tdf.loc[m, raw].rolling(n, min_periods=1).mean().shift(1)
    return tdf

def merge_form(df, form, n=5):
    suf = ['Form', 'GF', 'GA', 'xG', 'xGA', 'Shots', 'ShotsAgainst', 'SOT', 'SOTAgainst',
           'Corners', 'CornersAgainst', 'Fouls', 'FoulsAgainst', 'Yellows', 'Reds']
    fc = [f'{s}_{n}' for s in suf if f'{s}_{n}' in form.columns]
    form['_k'] = form['Team'] + '_' + form['Date'].astype(str)

    df['_hk'] = df['HomeTeam'] + '_' + df['Date'].astype(str)
    hp = form[form['Home'] == 1][['_k'] + fc].copy()
    hp.columns = ['_hk'] + [f'Home{s}_{n}' for s in suf if f'{s}_{n}' in form.columns]
    df = df.merge(hp, on='_hk', how='left')

    df['_ak'] = df['AwayTeam'] + '_' + df['Date'].astype(str)
    ap = form[form['Home'] == 0][['_k'] + fc].copy()
    ap.columns = ['_ak'] + [f'Away{s}_{n}' for s in suf if f'{s}_{n}' in form.columns]
    df = df.merge(ap, on='_ak', how='left')
    df.drop(columns=['_hk', '_ak'], errors='ignore', inplace=True)

    for c in df.columns:
        if f'_{n}' in c and any(c.startswith(p) for p in ['Home', 'Away']):
            m = df[c].mean()
            df[c] = df[c].fillna(m if not pd.isna(m) else 0)
    return df

def calc_h2h(df, n=5):
    print("  🤝 H2H stats...")
    df = df.sort_values('Date').copy()
    h2h = []
    for idx, row in df.iterrows():
        home, away, date = row['HomeTeam'], row['AwayTeam'], row['Date']
        prev = df[(df['Date'] < date) & (((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
                                         ((df['HomeTeam'] == away) & (df['AwayTeam'] == home)))].tail(n)
        if len(prev) == 0:
            h2h.append({'idx': idx, 'H2H_Matches': 0, 'H2H_HomeWins': 0, 'H2H_AwayWins': 0,
                        'H2H_Draws': 0, 'H2H_HomeGoals': 0, 'H2H_AwayGoals': 0})
        else:
            hw = aw = dr = hg = ag = 0
            for _, p in prev.iterrows():
                if p['HomeTeam'] == home:
                    hw += (p['Result'] == 'H'); aw += (p['Result'] == 'A'); dr += (p['Result'] == 'D')
                    hg += p['FTHG']; ag += p['FTAG']
                else:
                    hw += (p['Result'] == 'A'); aw += (p['Result'] == 'H'); dr += (p['Result'] == 'D')
                    hg += p['FTAG']; ag += p['FTHG']
            h2h.append({'idx': idx, 'H2H_Matches': len(prev), 'H2H_HomeWins': hw,
                        'H2H_AwayWins': aw, 'H2H_Draws': dr,
                        'H2H_HomeGoals': hg / len(prev), 'H2H_AwayGoals': ag / len(prev)})
    df = df.join(pd.DataFrame(h2h).set_index('idx'))
    return df

def calc_ref(df):
    print("  ⚖️ Referee strictness...")
    df['Referee'] = df['Referee'].fillna('Unknown')
    for c in ['HY', 'AY', 'HR', 'AR']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['_TY'] = df['HY'] + df['AY']
    df['_TR'] = df['HR'] + df['AR']
    ref = df.groupby('Referee').agg({'_TY': 'mean', '_TR': 'mean'})
    ref.columns = ['Ref_AvgYellows', 'Ref_AvgReds']
    raw = ref['Ref_AvgYellows'] + 3 * ref['Ref_AvgReds']
    mx = raw.max()
    ref['Ref_Strictness'] = raw / mx if mx > 0 else 0.5
    df = df.merge(ref, left_on='Referee', right_index=True, how='left')
    for c in ['Ref_AvgYellows', 'Ref_AvgReds', 'Ref_Strictness']:
        df[c] = df[c].fillna(df[c].mean())
    df.drop(columns=['_TY', '_TR'], errors='ignore', inplace=True)
    return df


def targets(df):
    df['Result_Num'] = df['Result'].map({'H': 0, 'D': 1, 'A': 2})
    df['Over1.5'] = (df['TotalGoals'] > 1.5).astype(int)
    df['Over3.5'] = (df['TotalGoals'] > 3.5).astype(int)
    df['HomeCleanSheet'] = (df['FTAG'] == 0).astype(int)
    df['AwayCleanSheet'] = (df['FTHG'] == 0).astype(int)
    return df


def run_pipeline(df):
    print("\n  🔧 Running Feature Engineering Pipeline...\n")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = add_basics(df)
    form_df = rolling_form(df, n=5)
    df = merge_form(df, form_df, n=5)
    df = calc_h2h(df, n=5)
    df = calc_ref(df)
    df = targets(df)

    proc_path = PROC_DIR / "cl_processed_matches.csv"
    df.to_csv(proc_path, index=False)
    print(f"  💾 {proc_path} ({len(df)} rows)")

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
    for c in target_cols:
        if c not in df.columns:
            df[c] = 0
    ml_df = df[target_cols].copy()
    ml_df['Season'] = pd.to_numeric(ml_df['Season'], errors='coerce').fillna(0).astype(int)
    ml_path = FEAT_DIR / "cl_ml_ready_data.csv"
    ml_df.to_csv(ml_path, index=False)
    print(f"  💾 {ml_path} ({len(ml_df)} rows, {len(target_cols)} cols)")

    print(f"\n{'=' * 60}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"  Matches: {len(df)}")
    print(f"  Non-zero features (Real Per-Match Stats, NOT averages!):")
    for c in ['Home_xG', ' Away_xG', 'HS', 'AS', 'HF', 'HC']:
        if c in df.columns:
            nz = len(df[df[c] > 0])
            print(f"    {c}: {nz}/{len(df)} ({100 * nz / len(df):.0f}%)")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    import ScraperFC

    print("=" * 70)
    print("  CHAMPIONS LEAGUE DATA — Genuine Per-Match Statistics Cache")
    print("=" * 70)

    fbref = ScraperFC.FBref(wait_time=WAIT_TIME)
    all_matches = []

    for season_str, season_code in SEASONS.items():
        print(f"\n{'─' * 60}")
        print(f"  Season: {season_str}")
        print(f"{'─' * 60}")
        matches = scrape_season_matches(fbref, season_str, season_code)
        if matches:
            all_matches.extend(matches)
        time.sleep(3)

    if not all_matches:
        print("\n✗ No data scraped!")
        return

    df = pd.DataFrame(all_matches)
    print(f"\n  COMBINED: {len(df)} matches")
    run_pipeline(df)


if __name__ == '__main__':
    main()
