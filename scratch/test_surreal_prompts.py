import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

load_dotenv()

import wisdom.config as cfg
from wisdom.providers import llm
from wisdom.agents.design import _IMAGE_PROMPT_TEMPLATE

def test_surreal_nature_prompts():
    style_name = "surreal_nature_vision"
    style_data = cfg.styles().get(style_name)
    style_desc = style_data.get("description", "").strip()
    
    test_quotes = [
        "In the depth of winter, I finally learned that there was in me an invincible summer.",
        "The soul that sees beauty may sometimes walk alone.",
        "Nature does not hurry, yet everything is accomplished."
    ]
    
    print(f"Testing Style: {style_name}\n")
    
    for i, text in enumerate(test_quotes):
        print(f"--- Test Case {i+1} ---")
        print(f"Quote: {text}")
        
        # Simulating the text zone instruction logic from design.py
        text_zone_instruction = "The top third of the frame will have #FFFFFF text overlaid on it. That area MUST be naturally clean, shadowed, or low-contrast in the scene itself — not bright or busy — so the text is legible."
        
        prompt = _IMAGE_PROMPT_TEMPLATE.format(
            text=text,
            style_name=style_name,
            style_description=style_desc,
            image_hint_block="",
            text_zone_instruction=text_zone_instruction,
        )
        
        try:
            image_prompt = llm.generate(prompt, role="creative_brief").strip()
            print(f"Generated Image Prompt:\n{image_prompt}\n")
        except Exception as e:
            print(f"Error generating prompt: {e}\n")

if __name__ == "__main__":
    test_surreal_nature_prompts()
