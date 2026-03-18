"""
Football Data Visualization & Analysis
Feature importance, team statistics, and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory for plots
OUTPUT_DIR = Path("notebooks/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load processed match data"""
    df = pd.read_csv("data/processed/processed_matches.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def plot_feature_importance(df, top_n=20):
    """Train a quick model to show feature importance"""
    print("Calculating feature importance...")
    
    # Select relevant features (no betting features - ethical choice)
    feature_cols = [
        'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
        'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
        'HomeForm_5', 'AwayForm_5', 'HomeGF_5', 'AwayGF_5',
        'HomeGA_5', 'AwayGA_5', 'H2H_Matches', 'H2H_HomeWins',
        'H2H_AwayWins', 'H2H_Draws', 'H2H_HomeGoals', 'H2H_AwayGoals'
    ]
    
    
    available_cols = [c for c in feature_cols if c in df.columns]
    
    # Prepare data
    X = df[available_cols].dropna()
    y = df.loc[X.index, 'Result']
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y_encoded)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': available_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True).tail(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 20 Most Important Features for Match Prediction', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved feature importance plot")
    return importance_df


def plot_correlation_heatmap(df):
    """Create correlation heatmap for key features"""
    print("Creating correlation heatmap...")
    
    # Select numeric columns
    numeric_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
                    'HC', 'AC', 'HY', 'AY', 'TotalGoals', 'GoalDiff']
    
    # Add form features if available
    form_cols = ['HomeForm_5', 'AwayForm_5', 'HomeGF_5', 'AwayGF_5']
    numeric_cols.extend([c for c in form_cols if c in df.columns])
    
    available_cols = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[available_cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, square=True, ax=ax,
                cbar_kws={'shrink': 0.8})
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved correlation heatmap")


def get_team_stats(df, team_name):
    """Calculate comprehensive statistics for a team"""
    # Home matches
    home_matches = df[df['HomeTeam'] == team_name].copy()
    # Away matches
    away_matches = df[df['AwayTeam'] == team_name].copy()
    
    if len(home_matches) == 0 and len(away_matches) == 0:
        return None
    
    stats = {
        'Team': team_name,
        'Total Matches': len(home_matches) + len(away_matches),
        'Home Matches': len(home_matches),
        'Away Matches': len(away_matches),
        
        # Home performance
        'Home Wins': len(home_matches[home_matches['Result'] == 'H']),
        'Home Draws': len(home_matches[home_matches['Result'] == 'D']),
        'Home Losses': len(home_matches[home_matches['Result'] == 'A']),
        'Home Goals For': home_matches['FTHG'].sum(),
        'Home Goals Against': home_matches['FTAG'].sum(),
        'Home Shots Avg': home_matches['HS'].mean() if 'HS' in home_matches.columns else np.nan,
        'Home Shots on Target Avg': home_matches['HST'].mean() if 'HST' in home_matches.columns else np.nan,
        
        # Away performance
        'Away Wins': len(away_matches[away_matches['Result'] == 'A']),
        'Away Draws': len(away_matches[away_matches['Result'] == 'D']),
        'Away Losses': len(away_matches[away_matches['Result'] == 'H']),
        'Away Goals For': away_matches['FTAG'].sum(),
        'Away Goals Against': away_matches['FTHG'].sum(),
        'Away Shots Avg': away_matches['AS'].mean() if 'AS' in away_matches.columns else np.nan,
        'Away Shots on Target Avg': away_matches['AST'].mean() if 'AST' in away_matches.columns else np.nan,
        
        # Cards
        'Home Yellow Cards': home_matches['HY'].sum() if 'HY' in home_matches.columns else np.nan,
        'Home Red Cards': home_matches['HR'].sum() if 'HR' in home_matches.columns else np.nan,
        'Away Yellow Cards': away_matches['AY'].sum() if 'AY' in away_matches.columns else np.nan,
        'Away Red Cards': away_matches['AR'].sum() if 'AR' in away_matches.columns else np.nan,
    }
    
    # Calculate derived stats
    total_wins = stats['Home Wins'] + stats['Away Wins']
    total_draws = stats['Home Draws'] + stats['Away Draws']
    total_losses = stats['Home Losses'] + stats['Away Losses']
    total_matches = stats['Total Matches']
    
    stats['Total Wins'] = total_wins
    stats['Total Draws'] = total_draws
    stats['Total Losses'] = total_losses
    stats['Win Rate'] = total_wins / total_matches if total_matches > 0 else 0
    stats['Points'] = total_wins * 3 + total_draws
    stats['Points Per Game'] = stats['Points'] / total_matches if total_matches > 0 else 0
    
    stats['Total Goals For'] = stats['Home Goals For'] + stats['Away Goals For']
    stats['Total Goals Against'] = stats['Home Goals Against'] + stats['Away Goals Against']
    stats['Goal Difference'] = stats['Total Goals For'] - stats['Total Goals Against']
    stats['Goals Per Game'] = stats['Total Goals For'] / total_matches if total_matches > 0 else 0
    stats['Goals Against Per Game'] = stats['Total Goals Against'] / total_matches if total_matches > 0 else 0
    
    return stats


def plot_team_comparison(df, team1, team2):
    """Create comparison visualizations for two teams"""
    print(f"Creating comparison: {team1} vs {team2}...")
    
    stats1 = get_team_stats(df, team1)
    stats2 = get_team_stats(df, team2)
    
    if stats1 is None or stats2 is None:
        print(f"  ✗ Could not find one or both teams in the data")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Win/Draw/Loss comparison
    ax1 = fig.add_subplot(2, 3, 1)
    categories = ['Wins', 'Draws', 'Losses']
    team1_values = [stats1['Total Wins'], stats1['Total Draws'], stats1['Total Losses']]
    team2_values = [stats2['Total Wins'], stats2['Total Draws'], stats2['Total Losses']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, team1_values, width, label=team1, color='#E63946')
    bars2 = ax1.bar(x + width/2, team2_values, width, label=team2, color='#1D3557')
    
    ax1.set_xlabel('Result')
    ax1.set_ylabel('Count')
    ax1.set_title('Win/Draw/Loss Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.bar_label(bars1, padding=3)
    ax1.bar_label(bars2, padding=3)
    
    # 2. Goals comparison
    ax2 = fig.add_subplot(2, 3, 2)
    categories = ['Goals For', 'Goals Against']
    team1_values = [stats1['Total Goals For'], stats1['Total Goals Against']]
    team2_values = [stats2['Total Goals For'], stats2['Total Goals Against']]
    
    x = np.arange(len(categories))
    bars1 = ax2.bar(x - width/2, team1_values, width, label=team1, color='#E63946')
    bars2 = ax2.bar(x + width/2, team2_values, width, label=team2, color='#1D3557')
    
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Goals')
    ax2.set_title('Goals Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.bar_label(bars1, padding=3)
    ax2.bar_label(bars2, padding=3)
    
    # 3. Performance metrics radar-style bar chart
    ax3 = fig.add_subplot(2, 3, 3)
    metrics = ['Win Rate', 'Goals/Game', 'PPG']
    team1_metrics = [stats1['Win Rate'], stats1['Goals Per Game'], stats1['Points Per Game']]
    team2_metrics = [stats2['Win Rate'], stats2['Goals Per Game'], stats2['Points Per Game']]
    
    x = np.arange(len(metrics))
    bars1 = ax3.bar(x - width/2, team1_metrics, width, label=team1, color='#E63946')
    bars2 = ax3.bar(x + width/2, team2_metrics, width, label=team2, color='#1D3557')
    
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Value')
    ax3.set_title('Performance Metrics', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Home vs Away performance
    ax4 = fig.add_subplot(2, 3, 4)
    categories = ['Home Wins', 'Away Wins']
    team1_values = [stats1['Home Wins'], stats1['Away Wins']]
    team2_values = [stats2['Home Wins'], stats2['Away Wins']]
    
    x = np.arange(len(categories))
    bars1 = ax4.bar(x - width/2, team1_values, width, label=team1, color='#E63946')
    bars2 = ax4.bar(x + width/2, team2_values, width, label=team2, color='#1D3557')
    
    ax4.set_xlabel('Location')
    ax4.set_ylabel('Wins')
    ax4.set_title('Home vs Away Wins', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.bar_label(bars1, padding=3)
    ax4.bar_label(bars2, padding=3)
    
    # 5. Shots comparison
    ax5 = fig.add_subplot(2, 3, 5)
    categories = ['Home Shots', 'Away Shots', 'Home SOT', 'Away SOT']
    team1_values = [stats1['Home Shots Avg'], stats1['Away Shots Avg'], 
                    stats1['Home Shots on Target Avg'], stats1['Away Shots on Target Avg']]
    team2_values = [stats2['Home Shots Avg'], stats2['Away Shots Avg'], 
                    stats2['Home Shots on Target Avg'], stats2['Away Shots on Target Avg']]
    
    x = np.arange(len(categories))
    bars1 = ax5.bar(x - width/2, team1_values, width, label=team1, color='#E63946')
    bars2 = ax5.bar(x + width/2, team2_values, width, label=team2, color='#1D3557')
    
    ax5.set_xlabel('Category')
    ax5.set_ylabel('Average per Match')
    ax5.set_title('Shots Statistics (Avg)', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, rotation=15)
    ax5.legend()
    
    # 6. Summary stats text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    TEAM COMPARISON SUMMARY
    ══════════════════════════════════════
    
    {team1:^20} vs {team2:^20}
    
    Total Matches:    {stats1['Total Matches']:^10}      {stats2['Total Matches']:^10}
    Total Points:     {stats1['Points']:^10}      {stats2['Points']:^10}
    Goal Difference:  {stats1['Goal Difference']:^+10}      {stats2['Goal Difference']:^+10}
    
    Win Rate:         {stats1['Win Rate']*100:^10.1f}%    {stats2['Win Rate']*100:^10.1f}%
    Goals/Game:       {stats1['Goals Per Game']:^10.2f}      {stats2['Goals Per Game']:^10.2f}
    Conceded/Game:    {stats1['Goals Against Per Game']:^10.2f}      {stats2['Goals Against Per Game']:^10.2f}
    
    Yellow Cards:     {stats1['Home Yellow Cards'] + stats1['Away Yellow Cards']:^10.0f}      {stats2['Home Yellow Cards'] + stats2['Away Yellow Cards']:^10.0f}
    Red Cards:        {stats1['Home Red Cards'] + stats1['Away Red Cards']:^10.0f}      {stats2['Home Red Cards'] + stats2['Away Red Cards']:^10.0f}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.suptitle(f'{team1} vs {team2} - Season 2022-2025 Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'comparison_{team1.replace(" ", "_")}_vs_{team2.replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved team comparison plot")
    return stats1, stats2


def plot_team_form_over_time(df, team_name):
    """Plot team's form evolution over time"""
    print(f"Creating form timeline for {team_name}...")
    
    # Get all matches
    home_matches = df[df['HomeTeam'] == team_name][['Date', 'Result', 'FTHG', 'FTAG', 'HomeForm_5']].copy()
    home_matches['Points'] = home_matches['Result'].map({'H': 3, 'D': 1, 'A': 0})
    home_matches['GF'] = home_matches['FTHG']
    home_matches['GA'] = home_matches['FTAG']
    home_matches['Form'] = home_matches['HomeForm_5']
    home_matches['Location'] = 'Home'
    
    away_matches = df[df['AwayTeam'] == team_name][['Date', 'Result', 'FTHG', 'FTAG', 'AwayForm_5']].copy()
    away_matches['Points'] = away_matches['Result'].map({'H': 0, 'D': 1, 'A': 3})
    away_matches['GF'] = away_matches['FTAG']
    away_matches['GA'] = away_matches['FTHG']
    away_matches['Form'] = away_matches['AwayForm_5']
    away_matches['Location'] = 'Away'
    
    all_matches = pd.concat([home_matches, away_matches]).sort_values('Date')
    
    if len(all_matches) == 0:
        print(f"  ✗ No matches found for {team_name}")
        return
    
    # Calculate cumulative points
    all_matches['Cumulative Points'] = all_matches['Points'].cumsum()
    all_matches['Match Number'] = range(1, len(all_matches) + 1)
    all_matches['Rolling Points'] = all_matches['Points'].rolling(5, min_periods=1).mean() * 3
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cumulative points over time
    ax1 = axes[0, 0]
    ax1.fill_between(all_matches['Match Number'], all_matches['Cumulative Points'], 
                      alpha=0.3, color='#2ecc71')
    ax1.plot(all_matches['Match Number'], all_matches['Cumulative Points'], 
             color='#27ae60', linewidth=2)
    ax1.set_xlabel('Match Number')
    ax1.set_ylabel('Points')
    ax1.set_title(f'{team_name} - Cumulative Points', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Form (5-match rolling average)
    ax2 = axes[0, 1]
    ax2.plot(all_matches['Match Number'], all_matches['Rolling Points'], 
             color='#3498db', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=1.5, color='#e74c3c', linestyle='--', label='Average (1.5 PPG)')
    ax2.axhline(y=2.0, color='#27ae60', linestyle='--', label='Good (2.0 PPG)')
    ax2.set_xlabel('Match Number')
    ax2.set_ylabel('Points Per Game (Rolling 5)')
    ax2.set_title(f'{team_name} - Form Over Time', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Goals scored vs conceded
    ax3 = axes[1, 0]
    ax3.bar(all_matches['Match Number'] - 0.2, all_matches['GF'], width=0.4, 
            label='Goals For', color='#2ecc71', alpha=0.7)
    ax3.bar(all_matches['Match Number'] + 0.2, all_matches['GA'], width=0.4, 
            label='Goals Against', color='#e74c3c', alpha=0.7)
    ax3.set_xlabel('Match Number')
    ax3.set_ylabel('Goals')
    ax3.set_title(f'{team_name} - Goals per Match', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Result distribution
    ax4 = axes[1, 1]
    result_counts = all_matches['Points'].map({3: 'Win', 1: 'Draw', 0: 'Loss'}).value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0, 0)
    
    if len(result_counts) > 0:
        wedges, texts, autotexts = ax4.pie(result_counts, labels=result_counts.index, 
                                            autopct='%1.1f%%', colors=colors[:len(result_counts)],
                                            explode=explode[:len(result_counts)],
                                            shadow=True, startangle=90)
        ax4.set_title(f'{team_name} - Result Distribution', fontweight='bold')
        
        # Add match count
        total = result_counts.sum()
        ax4.text(0, -1.3, f'Total: {total} matches', ha='center', fontsize=12)
    
    plt.suptitle(f'{team_name} Season Analysis (2022-2025)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'team_analysis_{team_name.replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved team form analysis")


def plot_league_overview(df):
    """Create league overview statistics"""
    print("Creating league overview...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Goals distribution
    ax1 = axes[0, 0]
    df['TotalGoals'].hist(bins=range(0, 12), ax=ax1, color='#3498db', edgecolor='white', alpha=0.7)
    ax1.axvline(df['TotalGoals'].mean(), color='red', linestyle='--', label=f'Mean: {df["TotalGoals"].mean():.2f}')
    ax1.set_xlabel('Total Goals in Match')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Total Goals per Match', fontweight='bold')
    ax1.legend()
    
    # 2. Result distribution by league
    ax2 = axes[0, 1]
    result_by_league = df.groupby(['League', 'Result']).size().unstack(fill_value=0)
    result_by_league.plot(kind='bar', ax=ax2, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax2.set_xlabel('League')
    ax2.set_ylabel('Count')
    ax2.set_title('Match Results by League', fontweight='bold')
    ax2.legend(title='Result')
    ax2.tick_params(axis='x', rotation=0)
    
    # 3. Home advantage
    ax3 = axes[1, 0]
    home_stats = {
        'Home Wins': (df['Result'] == 'H').sum(),
        'Draws': (df['Result'] == 'D').sum(),
        'Away Wins': (df['Result'] == 'A').sum()
    }
    colors = ['#27ae60', '#f39c12', '#e74c3c']
    wedges, texts, autotexts = ax3.pie(home_stats.values(), labels=home_stats.keys(), 
                                        autopct='%1.1f%%', colors=colors, 
                                        explode=(0.05, 0, 0), shadow=True)
    ax3.set_title('Home Advantage Analysis', fontweight='bold')
    
    # 4. Top scoring teams
    ax4 = axes[1, 1]
    home_goals = df.groupby('HomeTeam')['FTHG'].sum()
    away_goals = df.groupby('AwayTeam')['FTAG'].sum()
    total_goals = (home_goals.add(away_goals, fill_value=0)).sort_values(ascending=True).tail(10)
    
    total_goals.plot(kind='barh', ax=ax4, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(total_goals))))
    ax4.set_xlabel('Total Goals')
    ax4.set_title('Top 10 Goal Scoring Teams', fontweight='bold')
    
    plt.suptitle('League Overview - Premier League & La Liga (2022-2025)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'league_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved league overview")


def main():
    """Run all visualizations"""
    print("=" * 60)
    print("FOOTBALL DATA VISUALIZATION")
    print("=" * 60)
    print()
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} matches\n")
    
    # Create visualizations
    print("Creating visualizations...")
    print("-" * 40)
    
    # 1. Feature importance
    importance_df = plot_feature_importance(df)
    print("\nTop 10 Most Important Features:")
    print(importance_df.tail(10).to_string(index=False))
    print()
    
    # 2. Correlation heatmap
    plot_correlation_heatmap(df)
    
    # 3. League overview
    plot_league_overview(df)
    
    # 4. Team comparisons
    print()
    stats_barca, stats_manu = plot_team_comparison(df, 'Barcelona', 'Manchester United')
    
    # 5. Individual team analysis
    plot_team_form_over_time(df, 'Barcelona')
    plot_team_form_over_time(df, 'Manchester United')
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Print team stats
    print("\n" + "=" * 60)
    print("TEAM STATISTICS SUMMARY")
    print("=" * 60)
    
    for team, stats in [("Barcelona", stats_barca), ("Manchester United", stats_manu)]:
        print(f"\n{team.upper()}")
        print("-" * 40)
        print(f"  Total Matches: {stats['Total Matches']}")
        print(f"  Record: {stats['Total Wins']}W - {stats['Total Draws']}D - {stats['Total Losses']}L")
        print(f"  Points: {stats['Points']} ({stats['Points Per Game']:.2f} PPG)")
        print(f"  Goals: {stats['Total Goals For']} scored, {stats['Total Goals Against']} conceded")
        print(f"  Goal Difference: {stats['Goal Difference']:+d}")
        print(f"  Win Rate: {stats['Win Rate']*100:.1f}%")
        print(f"  Home Record: {stats['Home Wins']}W - {stats['Home Draws']}D - {stats['Home Losses']}L")
        print(f"  Away Record: {stats['Away Wins']}W - {stats['Away Draws']}D - {stats['Away Losses']}L")


if __name__ == "__main__":
    main()
