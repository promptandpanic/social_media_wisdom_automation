from PIL import ImageFont
import os

font_path = "assets/fonts/inter.ttf"
try:
    font = ImageFont.truetype(font_path, 40)
    print("SUCCESS: Inter font loaded.")
except Exception as e:
    print(f"FAILURE: {e}")
