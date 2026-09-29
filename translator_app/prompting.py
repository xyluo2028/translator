from __future__ import annotations

from typing import Any, Literal

from translator_app.models import TranslateRequest


def build_system_prompt(request: TranslateRequest) -> str:
    tone_line = f"Tone/style: {request.tone}."
    if request.tone_instructions:
        tone_line += f" Additional instructions: {request.tone_instructions!r}."

    base = (
        "You are a high-precision translation engine.\n"
        "- Preserve meaning, numbers, and proper nouns.\n"
        "- Preserve line breaks and punctuation when reasonable.\n"
        "- Output JSON only (no code fences, no extra text).\n"
        "- Always include all required keys; use null when unknown.\n"
    )

    if request.mode == "dictionary":
        return (
            base
            + "\n"
            + tone_line
            + "\n"
            + "Task: Return dictionary-style entries with multiple senses.\n"
            + "Constraints:\n"
            + "- Provide up to 2 parts-of-speech and up to 3 senses each.\n"
            + "- Keep example sentences short.\n"
            + "Required JSON keys:\n"
            + '- term (string)\n'
            + '- entries (array of objects)\n'
            + '  - pos (string|null)\n'
            + '  - senses (array)\n'
            + '    - meaning (string)\n'
            + '    - example_source (string|null)\n'
            + '    - example_target (string|null)\n'
            + '    - usage_notes (string|null)\n'
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
        + "Required JSON keys:\n"
        + '- translation (string)\n'
        + '- alternatives (array of strings|null)\n'
        + '- notes (string|null)\n'
        + '- detected_source_lang (string|null)\n'
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


# --- Translation-only models (Hunyuan MT, TranslateGemma) -------------------------------------
# These models are fine-tuned on one fixed prompt each and reply with the bare translation, so they
# get their official template instead of the JSON schema prompt above.

PromptStyle = Literal["json", "hunyuan", "translategemma"]

# code -> (English name, Chinese name). Chinese names are used by Hunyuan's ZH<=>XX template.
LANGUAGES: dict[str, tuple[str, str]] = {
    "EN": ("English", "英语"),
    "ZH": ("Chinese", "中文"),
    "ZH-TW": ("Traditional Chinese", "繁体中文"),
    "JA": ("Japanese", "日语"),
    "KO": ("Korean", "韩语"),
    "FR": ("French", "法语"),
    "DE": ("German", "德语"),
    "ES": ("Spanish", "西班牙语"),
    "PT": ("Portuguese", "葡萄牙语"),
    "IT": ("Italian", "意大利语"),
    "RU": ("Russian", "俄语"),
    "AR": ("Arabic", "阿拉伯语"),
    "TH": ("Thai", "泰语"),
    "VI": ("Vietnamese", "越南语"),
    "ID": ("Indonesian", "印尼语"),
    "HI": ("Hindi", "印地语"),
}


def prompt_style_for(model: str) -> PromptStyle:
    name = model.lower()
    if "hy-mt" in name or "hunyuan-mt" in name:
        return "hunyuan"
    if "translategemma" in name:
        return "translategemma"
    return "json"


def detect_language(text: str) -> str | None:
    """Cheap script-based guess, enough to pick a prompt template. Returns a LANGUAGES code or None."""
    kana = hangul = han = latin = 0
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:
            kana += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            hangul += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            han += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    if kana:
        return "JA"
    if hangul > han:
        return "KO"
    if han:
        return "ZH"
    if latin:
        return "EN"
    return None


def _lang_name(code: str, *, chinese: bool = False) -> str:
    names = LANGUAGES.get(code.upper())
    if names is None:
        return code
    return names[1] if chinese else names[0]


def build_plain_prompt(request: TranslateRequest, *, style: PromptStyle, source_lang: str | None) -> str:
    text = request.text.strip()
    tgt = request.target_lang.upper()

    if style == "hunyuan":
        if tgt.startswith("ZH") or (source_lang or "").upper().startswith("ZH"):
            return (
                f"将以下文本翻译为{_lang_name(tgt, chinese=True)}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
                f"{text}"
            )
        return f"Translate the following segment into {_lang_name(tgt)}, without additional explanation.\n\n{text}"

    src = (source_lang or "").upper()
    src_name = _lang_name(src) if src else "source language"
    src_label = f"{src_name} ({src.lower()})" if src else src_name
    tgt_name = _lang_name(tgt)
    return (
        f"You are a professional {src_label} to {tgt_name} ({tgt.lower()}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {src_name} text "
        f"while adhering to {tgt_name} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tgt_name} translation, without any additional explanations or commentary. "
        f"Please translate the following {src_name} text into {tgt_name}:\n\n\n"
        f"{text}"
    )


def build_plain_messages(
    request: TranslateRequest, *, style: PromptStyle, source_lang: str | None
) -> tuple[list[dict[str, Any]], bool]:
    """Chat messages + add_generation_prompt for Hugging Face chat templates of translation-only models."""
    if style == "translategemma":
        # TranslateGemma's HF chat template takes structured content with ISO language codes.
        if not source_lang:
            raise ValueError("TranslateGemma needs a source language; pick one instead of auto-detect.")
        content = {
            "type": "text",
            "source_lang_code": _iso_code(source_lang),
            "target_lang_code": _iso_code(request.target_lang),
            "text": request.text.strip(),
        }
        return [{"role": "user", "content": [content]}], True
    # Hunyuan MT: one user turn, no system prompt, and add_generation_prompt=False per Tencent's model card.
    prompt = build_plain_prompt(request, style=style, source_lang=source_lang)
    return [{"role": "user", "content": prompt}], False


def _iso_code(code: str) -> str:
    lang, _, region = code.partition("-")
    return f"{lang.lower()}-{region.upper()}" if region else lang.lower()
