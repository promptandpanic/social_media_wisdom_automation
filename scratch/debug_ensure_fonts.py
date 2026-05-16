import logging
import requests
from pathlib import Path
from wisdom.composers.card import _FONT_URLS, FONTS_DIR

# Set up logging to stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wisdom.composers.card")

def debug_ensure():
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for key, (filename, url) in _FONT_URLS.items():
        path = FONTS_DIR / filename
        if not path.exists():
            print(f"DEBUG: Downloading {key} from {url}")
            try:
                r = requests.get(url, timeout=30)
                print(f"DEBUG: Status {r.status_code}")
                if r.status_code == 200:
                    path.write_bytes(r.content)
                    print(f"DEBUG: Success {filename}")
                else:
                    print(f"DEBUG: Failed {r.status_code}")
            except Exception as e:
                print(f"DEBUG: Exception {e}")
        else:
            print(f"DEBUG: {key} already exists")

if __name__ == "__main__":
    debug_ensure()
