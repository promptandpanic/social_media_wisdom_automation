import os
import sys
from dotenv import load_dotenv

sys.path.append("/Users/deepakrout/Documents/codebase/personal/social_media_wisdom_automation")
load_dotenv("/Users/deepakrout/Documents/codebase/personal/social_media_wisdom_automation/.env")

from wisdom.providers import llm

try:
    content, provider = llm.generate("Hello, who are you?", role="quote_generation")
    print(f"Success! Provider: {provider}\nContent: {content}")
except Exception as e:
    print(f"Error: {e}")
