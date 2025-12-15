# LLM Translator (WIP)

Build a practical translator app backed by an LLM (e.g. `"hello" -> "你好"`), with tone controls, dictionary-style lookup, and fast “retry / regenerate” when the result isn’t satisfying.

## Quickstart (M1)

Prereqs: Python 3.13+, Ollama running locally (`ollama serve`), and a model pulled (e.g. `ollama pull gpt-oss:20b`).

1) Edit `config.toml` (or keep defaults).

2) Run:

```bash
python3 translate.py "hello" --to ZH
python3 translate.py "How are you?" --to JA --tone polite
python3 translate.py "run" --mode dictionary --to EN
python3 translate.py "hello" --to ZH --rerun more_literal
```

## Goals

- **Fast translation** between languages with optional **auto-detect**.
- **Tone / style presets** (casual, formal, polite, spoken, business, etc).
- **Dictionary mode** for single words/phrases (multiple senses, POS, examples).
- **Refresh / rerun** to regenerate (optionally with “more literal / more natural”).
- **Provider-agnostic**: local-first (Ollama), plus API providers via key.

## Non-goals (for MVP)

- Perfect human-level translation for every niche domain.
- Full offline bilingual dictionary database (start LLM-based; add later if needed).
- Real-time collaboration / multi-user permissions.

## Product Draft (UX)

**Main screen**
- Input textarea
- Source language: `Auto` or fixed (e.g. `EN`)
- Target language: required (e.g. `ZH`, `JA`)
- Mode: `Translate` | `Dictionary`
- Tone: preset dropdown + “custom instructions” (optional)
- Output panel with:
  - primary result
  - alternatives (optional)
  - notes (optional)
  - buttons: `Copy`, `Refresh`, `More literal`, `More natural`

**History (optional for MVP but recommended)**
- recent translations
- click to restore input/settings

## Blueprint (Architecture)

Keep the system modular so the UI is a thin client and the “translation engine” can be reused by CLI/API.

### Components

- **Core engine (library)**
  - request validation + normalization
  - language detection (optional)
  - prompt builder (mode + tone + constraints)
  - LLM provider abstraction
  - output parsing (structured JSON)
  - post-processing (cleanup, formatting)
  - persistence hooks (history/cache)

- **Provider adapters**
  - `ollama` (local) as default
  - `openai` / `anthropic` / others (API-key based), optional
  - common interface: `generate(prompt, params) -> text`

- **API server (optional but recommended)**
  - endpoints: `/translate`, `/detect`, `/history`
  - handles auth/storage on behalf of UI

- **UI**
  - web app (local server) or desktop wrapper (later)

### Data flow (Translate)

1. UI sends `TranslateRequest`
2. Server/core normalizes (trim, max length, etc)
3. If source is `Auto`, detect source language
4. Build prompt from: mode + languages + tone + extra instructions + output schema
5. Call provider (Ollama/API)
6. Parse model output into a structured `TranslateResult`
7. Persist (optional): history + cache
8. Return result to UI

## Data contracts (suggested)

### TranslateRequest

- `text`: string
- `source_lang`: string (`"auto"` allowed)
- `target_lang`: string
- `mode`: `"translate" | "dictionary"`
- `tone`: preset name (e.g. `"casual"`) or `"custom"`
- `tone_instructions`: optional string
- `explain_lang`: language for explanations/notes (e.g. `"EN"`)
- `rerun`: optional object to guide regeneration (e.g. `{ "style": "more_literal" }`)

### TranslateResult

- `translation`: string
- `alternatives`: string[] (optional)
- `notes`: string (optional)
- `detected_source_lang`: string (when source is auto)
- `provider`: string, `model`: string, `latency_ms`: number (optional)

### DictionaryResult (when `mode="dictionary"`)

- `term`: string
- `entries`: array of:
  - `pos`: string (noun/verb/adj/…)
  - `senses`: array of:
    - `meaning`: string
    - `example_source`: string (optional)
    - `example_target`: string (optional)
    - `usage_notes`: string (optional)

## Prompting strategy (practical + parseable)

Prefer **structured output** so the UI can render consistently.

- Use a “system-style” instruction: translate accurately, follow tone, no extra text.
- Require **JSON only** output for the selected schema.
- Add guardrails:
  - keep proper nouns unchanged unless requested
  - preserve punctuation and line breaks when reasonable
  - do not invent facts; if ambiguous, add notes

## Config (suggested `config.toml`)

```toml
provider = "ollama" # or "openai"

[ollama]
host = "http://localhost:11434"
model = "qwen2.5:7b"

[openai]
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"

[app]
default_target_lang = "ZH"
default_tone = "neutral"
history_db = "translator.db"
```

## Tech choices (recommended to start)

If you want fastest iteration:
- **Python** core + **FastAPI** for local API
- Simple web UI (React/Vite) *or* even start with a CLI and add UI later

If you want one-binary distribution later:
- Core stays provider-agnostic; add a desktop wrapper (e.g. Tauri/Electron) after MVP.

## Doable milestone plan

### M0 — Repo setup (0.5 day)
- Pick language/runtime (recommended: Python)
- Add basic project layout + `config.toml` parsing
- Add a single “hello world” command that prints config/provider info

### M1 — Core translate engine (1–2 days)
- Define request/result schemas (Translate + Dictionary)
- Implement prompt builder for `translate` mode
- Implement provider adapter for **Ollama**
- Add “refresh” by re-calling with new seed or rerun hint

### M2 — CLI (0.5–1 day)
- `translate "text" --to ZH --from auto --tone casual`
- `dict "term" --to EN`
- Print structured output + a human-readable view

### M3 — Local API + minimal UI (1–2 days)
- FastAPI endpoints: `/translate`, `/history` (optional)
- Minimal UI: input, language selectors, tone dropdown, output, refresh

### M4 — Dictionary mode (1 day)
- Prompt + JSON schema for dictionary output
- UI rendering for multiple senses/examples

### M5 — Tones + presets (0.5–1 day)
- Preset library (casual/formal/polite/spoken/business)
- “Custom instructions” field merged safely into prompt

### M6 — Quality + polish (1–2 days)
- History + cache (SQLite)
- Basic eval set + regression checks (hand-curated examples)
- Guardrails: max length, profanity filter option, PII redaction option (optional)

## Open questions

- Desktop vs web vs CLI-first?
- Do we need streaming output (token-by-token) for responsiveness?
- Should we support a user glossary/terminology list early (very useful for names/brands)?
