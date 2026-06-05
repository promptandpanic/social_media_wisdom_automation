import sys
from wisdom.providers.image import PollinationsProvider

prompt = "A majestic lion standing on a cliff at sunset, cinematic lighting, 8k"
print("Generating image using Pollinations...")
provider = PollinationsProvider()
try:
    image_bytes = provider.generate(prompt)
    with open("output/test_pollinations.jpg", "wb") as f:
        f.write(image_bytes)
    print("Success! Image saved to output/test_pollinations.jpg")
except Exception as e:
    print(f"Error: {e}")
