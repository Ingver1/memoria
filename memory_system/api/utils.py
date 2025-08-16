from fastapi import HTTPException


def validate_text_length(text: str, max_len: int) -> None:
    """Ensure text does not exceed the maximum allowed length."""
    if len(text) > max_len:
        raise HTTPException(status_code=400, detail="text exceeds maximum length")
