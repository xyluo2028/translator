from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Mode = Literal["translate", "dictionary", "enhance", "relatives"]
RerunStyle = Literal["retry", "more_literal", "more_natural"]


@dataclass(frozen=True)
class RerunHint:
    style: RerunStyle


@dataclass(frozen=True)
class SpellingCorrection:
    word: str
    suggestion: str
    alternatives: list[str] = field(default_factory=list)
    # Half-open Unicode code point offsets in SpellingFix.original.
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class SpellingFix:
    """Source text was corrected before translating ("did you mean ...?")."""

    original: str
    corrected: str
    corrections: list[SpellingCorrection]


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
    # Overrides the provider's configured model for this request.
    model: str | None = None
    # Correct likely misspellings after resolving the source language (needs the optional `text` extra).
    spellcheck: bool = True
    # Writing context used by Enhance; other modes ignore it.
    scenario: str = "general"


@dataclass(frozen=True)
class TranslateResult:
    translation: str
    alternatives: list[str] | None = None
    notes: str | None = None
    detected_source_lang: str | None = None
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    spelling: SpellingFix | None = None


@dataclass(frozen=True)
class PromptEnhanceResult:
    # Keep the existing JSON field for CLI clients and saved web history.
    enhanced_prompt: str
    clarifications: list[str] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    spelling: SpellingFix | None = None


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
class Pronunciation:
    label: str
    ipa: str


@dataclass(frozen=True)
class DictionaryResult:
    term: str
    entries: list[DictionaryEntry]
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    spelling: SpellingFix | None = None
    pronunciations: list[Pronunciation] = field(default_factory=list)


@dataclass(frozen=True)
class RelatedTerm:
    term: str
    pos: str
    meaning: str


@dataclass(frozen=True)
class RelativesResult:
    term: str
    derivations: list[RelatedTerm] = field(default_factory=list)
    synonyms: list[RelatedTerm] = field(default_factory=list)
    antonyms: list[RelatedTerm] = field(default_factory=list)
    notes: str | None = None
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    spelling: SpellingFix | None = None
