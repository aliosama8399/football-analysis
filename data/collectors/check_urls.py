import requests

BASE = "https://www.football-data.co.uk/mmz4281/2223"
PATTERNS = [
    "CL.csv",
    "E1.csv",
    "EC.csv", # We know this is Conference
    "SC0.csv",
    "D1.csv" # Known good
]

print("Checking URLs...")
for p in PATTERNS:
    url = f"{BASE}/{p}"
    try:
        r = requests.head(url)
        print(f"{r.status_code} : {url}")
    except Exception as e:
        print(f"Error {url}: {e}")
