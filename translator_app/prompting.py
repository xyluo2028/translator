from __future__ import annotations

from translator_app.models import TranslateRequest


def build_system_prompt(request: TranslateRequest) -> str:
    tone_line = f"Tone/style: {request.tone}."
    if request.tone_instructions:
        tone_line += f" Additional instructions: {request.tone_instructions!r}."

    base = (
        "You are a high-precision translation engine.\n"
        "- Preserve meaning, numbers, and proper nouns.\n"
        "- Preserve line breaks and punctuation when reasonable.\n"
        "- Do not add commentary outside the required JSON.\n"
    )

    if request.mode == "dictionary":
        return (
            base
            + "\n"
            + tone_line
            + "\n"
            + "Task: Return dictionary-style entries with multiple senses.\n"
            + 'Output: JSON only, matching this schema:\n'
            + '{\n'
            + '  "term": string,\n'
            + '  "entries": [\n'
            + "    {\n"
            + '      "pos": string | null,\n'
            + '      "senses": [\n'
            + "        {\n"
            + '          "meaning": string,\n'
            + '          "example_source": string | null,\n'
            + '          "example_target": string | null,\n'
            + '          "usage_notes": string | null\n'
            + "        }\n"
            + "      ]\n"
            + "    }\n"
            + "  ]\n"
            + "}\n"
        )

    rerun_line = ""
    if request.rerun:
        if request.rerun.style == "more_literal":
            rerun_line = "Regeneration hint: make it more literal (closer to source wording)."
        elif request.rerun.style == "more_natural":
            rerun_line = "Regeneration hint: make it more natural (native phrasing) while preserving meaning."
        else:
            rerun_line = "Regeneration hint: try a different valid translation."

    return (
        base
        + "\n"
        + tone_line
        + ("\n" + rerun_line if rerun_line else "")
        + "\n"
        + 'Output: JSON only, matching this schema:\n'
        + '{\n'
        + '  "translation": string,\n'
        + '  "alternatives": string[] | null,\n'
        + '  "notes": string | null,\n'
        + '  "detected_source_lang": string | null\n'
        + "}\n"
        + "If the source language is ambiguous, set detected_source_lang to null and explain briefly in notes.\n"
        + "Notes language: "
        + request.explain_lang
        + ".\n"
    )


def build_user_prompt(request: TranslateRequest) -> str:
    if request.mode == "dictionary":
        return (
            f"Explain and translate as a dictionary entry.\n"
            f"Target language for meanings/examples: {request.target_lang}.\n"
            f"Term: {request.text.strip()}\n"
        )

    src = request.source_lang
    tgt = request.target_lang
    return (
        "Translate the text.\n"
        f"Source language: {src}.\n"
        f"Target language: {tgt}.\n"
        "Text:\n"
        f"{request.text}\n"
    )

