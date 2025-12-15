from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Mode = Literal["translate", "dictionary"]
RerunStyle = Literal["retry", "more_literal", "more_natural"]


@dataclass(frozen=True)
class RerunHint:
    style: RerunStyle


@dataclass(frozen=True)
class TranslateRequest:
    text: str
    source_lang: str
    target_lang: str
    mode: Mode = "translate"
    tone: str = "neutral"
    tone_instructions: str | None = None
    explain_lang: str = "EN"
    rerun: RerunHint | None = None
    seed: int | None = None
    temperature: float = 0.2


@dataclass(frozen=True)
class TranslateResult:
    translation: str
    alternatives: list[str] | None = None
    notes: str | None = None
    detected_source_lang: str | None = None
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None


@dataclass(frozen=True)
class DictionarySense:
    meaning: str
    example_source: str | None = None
    example_target: str | None = None
    usage_notes: str | None = None


@dataclass(frozen=True)
class DictionaryEntry:
    pos: str | None
    senses: list[DictionarySense]


@dataclass(frozen=True)
class DictionaryResult:
    term: str
    entries: list[DictionaryEntry]
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None

