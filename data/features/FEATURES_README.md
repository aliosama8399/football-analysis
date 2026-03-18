# Football Dataset Feature Documentation

## Overview
This document describes all features in the processed football dataset.
Dataset: `data/processed/processed_matches.csv`

---

## Match Identification

| Feature | Type | Description |
|---------|------|-------------|
| `Date` | datetime | Match date |
| `HomeTeam` | string | Home team name (standardized) |
| `AwayTeam` | string | Away team name (standardized) |
| `League` | string | League identifier (Premier_League, La_Liga) |
| `Season` | string | Season identifier (e.g., "2223") |

---

## Match Result Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `FTHG` | int | 0-10 | Full-time home goals |
| `FTAG` | int | 0-10 | Full-time away goals |
| `FTR` | string | H/D/A | Full-time result |
| `HTHG` | int | 0-6 | Half-time home goals |
| `HTAG` | int | 0-6 | Half-time away goals |
| `Result` | string | H/D/A | Match result (standardized) |
| `TotalGoals` | int | 0-15 | FTHG + FTAG |
| `GoalDiff` | int | -8 to 8 | FTHG - FTAG |

---

## Match Statistics

| Feature | Type | Description |
|---------|------|-------------|
| `HS` | int | Home team shots |
| `AS` | int | Away team shots |
| `HST` | int | Home team shots on target |
| `AST` | int | Away team shots on target |
| `HF` | int | Home team fouls |
| `AF` | int | Away team fouls |
| `HC` | int | Home team corners |
| `AC` | int | Away team corners |
| `HY` | int | Home team yellow cards |
| `AY` | int | Away team yellow cards |
| `HR` | int | Home team red cards |
| `AR` | int | Away team red cards |

---

## Form Features (Rolling 5-match)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `HomeForm_5` | float | 0-3 | Home team avg points last 5 games |
| `AwayForm_5` | float | 0-3 | Away team avg points last 5 games |
| `HomeGF_5` | float | 0-5 | Home team avg goals scored last 5 |
| `HomeGA_5` | float | 0-5 | Home team avg goals conceded last 5 |
| `AwayGF_5` | float | 0-5 | Away team avg goals scored last 5 |
| `AwayGA_5` | float | 0-5 | Away team avg goals conceded last 5 |

**Note**: First 5 games of season have incomplete form data (uses available matches).

---

## Head-to-Head Features

| Feature | Type | Description |
|---------|------|-------------|
| `H2H_Matches` | int | Number of previous meetings (up to 5) |
| `H2H_HomeWins` | int | Home team wins in H2H |
| `H2H_AwayWins` | int | Away team wins in H2H |
| `H2H_Draws` | int | Draws in H2H |
| `H2H_HomeGoals` | int | Total home goals in H2H |
| `H2H_AwayGoals` | int | Total away goals in H2H |

**Note**: Zeros indicate teams haven't met in the dataset history.

---

## Referee Strictness Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `Ref_Strictness` | float | 0-1 | Referee strictness score (normalized) |
| `Ref_AvgYellows` | float | 0-8 | Referee avg yellow cards per match |
| `Ref_AvgReds` | float | 0-1 | Referee avg red cards per match |

**Calculation**: `Strictness = (AvgYellows + 3*AvgReds) / max_value`

Higher values = stricter referee. Missing referees use league average.

---

## Target Variables (For ML)

| Feature | Type | Description |
|---------|------|-------------|
| `Result_Num` | int | 0=Home Win, 1=Draw, 2=Away Win |
| `Over1.5` | binary | 1 if TotalGoals > 1.5 |
| `Over2.5` | binary | 1 if TotalGoals > 2.5 |
| `Over3.5` | binary | 1 if TotalGoals > 3.5 |
| `BTTS` | binary | 1 if both teams scored |
| `HomeCleanSheet` | binary | 1 if away team scored 0 |
| `AwayCleanSheet` | binary | 1 if home team scored 0 |

---

## Data Quality Notes

### Missing Values
- **Form features (first 5 games)**: Incomplete rolling windows at season start
- **H2H features (zeros)**: Teams without prior meetings in dataset
- **xG features**: Available when Understat data is loaded

### Removed Features
- **Betting odds**: Removed for ethical reasons (B365, BW, IW, etc.)
- **Referee name**: Replaced with strictness score only

### Future Features (Planned)
- **Injury count**: Number of injured players before match
- **Team strength**: Based on cumulative season performance

---

## Usage Example

```python
import pandas as pd

# Load data
df = pd.read_csv('data/processed/processed_matches.csv')

# Features for ML
features = ['HomeForm_5', 'AwayForm_5', 'H2H_Matches', 
            'Ref_Strictness', 'HomeGF_5', 'AwayGF_5']
target = 'Result_Num'

X = df[features].fillna(0)
y = df[target]
```

---

*Last updated: 2026-01-13*
