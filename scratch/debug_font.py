import requests
from pathlib import Path

url = "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/static/Inter-Regular.ttf"
filename = "assets/fonts/inter.ttf"

print(f"Downloading {url}...")
try:
    r = requests.get(url, timeout=30)
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        Path(filename).write_bytes(r.content)
        print(f"Saved to {filename} ({len(r.content)} bytes)")
    else:
        print(f"Error: {r.text[:200]}")
except Exception as e:
    print(f"Exception: {e}")
