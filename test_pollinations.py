import sys
import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.environ.get("POLLINATIONS_API_KEY")
if not api_key:
    print("No key found")
    sys.exit(1)

print(f"Key loaded: {api_key[:5]}...")

url = f"https://image.pollinations.ai/prompt/A%20lion?width=1080&height=1920&key={api_key}"
resp = requests.get(url)
print(f"Query Param Test: {resp.status_code}")

url2 = f"https://image.pollinations.ai/prompt/A%20lion?width=1080&height=1920"
headers = {"Authorization": f"Bearer {api_key}"}
resp2 = requests.get(url2, headers=headers)
print(f"Header Test: {resp2.status_code}")
