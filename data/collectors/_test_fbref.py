"""Test soccerdata for Champions League data."""
import soccerdata as sd

print("Available leagues:")
fb = sd.FBref(leagues='Champions League', seasons='2023-2024')
print(type(fb))

print("\n=== SCHEDULE ===")
try:
    sched = fb.read_schedule()
    print(f"Shape: {sched.shape}")
    print(f"Columns: {list(sched.columns)}")
    print(sched.head(3))
except Exception as e:
    print(f"Schedule error: {e}")

print("\n=== TEAM SEASON STATS (shooting) ===")
try:
    shooting = fb.read_team_season_stats(stat_type='shooting')
    print(f"Shape: {shooting.shape}")
    print(f"Columns: {list(shooting.columns)[:15]}")
    print(shooting.head(3))
except Exception as e:
    print(f"Shooting error: {e}")

print("\n=== TEAM MATCH STATS ===")
try:
    match_stats = fb.read_team_match_stats(stat_type='schedule')
    print(f"Shape: {match_stats.shape}")
    print(f"Columns: {list(match_stats.columns)}")
    print(match_stats.head(3))
except Exception as e:
    print(f"Match stats error: {e}")
