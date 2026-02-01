import re

def clean_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', ' ', text)

    # Remove URLs
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        ' ',
        text
    )

    # Remove common navigation / footer noise
    text = re.sub(
        r'(Skip navigation|Sign in|Send feedback|Privacy|Terms|Manage cookies)',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # Keep punctuation useful for LLMs & job descriptions
    text = re.sub(r'[^a-zA-Z0-9 .,/:;+()-]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
