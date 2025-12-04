"""Configuration file for Shopping Agent"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for shopping agent"""

    # API Configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # Latest Claude model

    # Browser Configuration
    HEADLESS_MODE = os.getenv("HEADLESS_MODE", "False").lower() == "true"
    BROWSER_TYPE = os.getenv("BROWSER_TYPE", "chromium")
    TIMEOUT = int(os.getenv("TIMEOUT", "30000"))

    # Screenshot Configuration
    SCREENSHOT_DIR = Path(os.getenv("SCREENSHOT_DIR", "screenshots"))
    SCREENSHOT_DIR.mkdir(exist_ok=True)

    # Shopping Sites Configuration
    SUPPORTED_SITES = {
        "calvinklein_us": "https://www.calvinklein.us/",
        "calvinklein_ca": "https://www.calvinklein.ca/",
        "tommy_us": "https://usa.tommy.com/",
        "tommy_ca": "https://ca.tommy.com/en"
    }

    # Agent Configuration
    MAX_RETRIES = 3
    AGENT_TEMPERATURE = 0.7
    MAX_TOKENS = 4096

    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required. Please set it in .env file")
        return True
