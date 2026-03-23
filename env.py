"""Load .env and map lowercase keys to uppercase for AWS SDK compatibility."""

import os

_AWS_KEYS = [
    "aws_region",
    "aws_default_region",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "anthropic_api_key",
]


def load_env():
    """Load .env file and normalize AWS keys to uppercase."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    for key in _AWS_KEYS:
        val = os.environ.get(key)
        if val and not os.environ.get(key.upper()):
            os.environ[key.upper()] = val
