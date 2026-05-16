import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
from wisdom.agents import design
from wisdom.schemas import Quote, ThemeConfig

async def test():
    theme = ThemeConfig(
        key="wisdom",
        name="Life Wisdom",
        format="reel",
        max_words=18,
        platforms=["instagram"],
        hashtags=["#wisdom"]
    )
    quote = Quote(
        text="The only way to do great work is to love what you do.",
        author="Steve Jobs",
        highlight="love what you do",
        source="real_author"
    )
    state = {
        "theme_key": "wisdom",
        "theme": theme,
        "quote": quote,
        "recent_styles": [],
        "_chosen_style": "surreal_cosmic_minimalism"
    }
    
    # Force gemini
    os.environ["LLM_PROVIDER_ORDER"] = "gemini"
    
    new_state = design.generate_brief(state)
    brief = new_state["brief"]
    print(f"\nSTYLE: {brief.style}")
    print(f"FONT: {brief.font}")
    print(f"PROMPT: {brief.image_prompt}")

if __name__ == "__main__":
    asyncio.run(test())
