import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda _: None


def load_secrets():
    """Load environment variables from .env file"""
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
