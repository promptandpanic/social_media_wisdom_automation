import os
import requests
from dotenv import load_dotenv

load_dotenv("/Users/deepakrout/Documents/codebase/personal/social_media_wisdom_automation/.env")

token = os.environ.get("FACEBOOK_ACCESS_TOKEN")
page_id = os.environ.get("FACEBOOK_PAGE_ID")

print(f"Page ID: {page_id}")

r = requests.get(
    f"https://graph.facebook.com/v19.0/debug_token",
    params={"input_token": token, "access_token": token}
)
print("Debug Token:\n", r.json())

r2 = requests.get(
    f"https://graph.facebook.com/v19.0/me/accounts",
    params={"access_token": token}
)
print("My Pages:\n", r2.json())
