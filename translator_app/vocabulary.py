"""Input limits for vocabulary-only features, shared by the CLI and server."""
MAX_RELATIVES_WORDS = 5
MAX_RELATIVES_CHARS = 60
RELATIVES_INPUT_HINT = "Enter one word or a short phrase (up to 5 words and 60 characters), without sentence punctuation."


def validate_relatives_text(text: str) -> None:
    value = text.strip()
    if (
        not value or len(value) > MAX_RELATIVES_CHARS or len(value.split()) > MAX_RELATIVES_WORDS
        or any(char in value for char in "\n\r.!?。！？;；")
        or not any(char.isalpha() for char in value)
    ):
        raise ValueError(f"Relatives works with words and short phrases only. {RELATIVES_INPUT_HINT}")
