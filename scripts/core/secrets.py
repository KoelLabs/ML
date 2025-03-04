import os
from dotenv import load_dotenv


def load_secrets():
    """Load environment variables from .env file"""
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
