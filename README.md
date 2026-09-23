# LLM Translator

A portable, local-first translator: runs entirely on your own machine with open models, served through
**Ollama** or **Hugging Face Transformers**. Use it from a local web UI or the command line.

- Translate between English, Chinese, Japanese, Korean and a dozen other languages, with auto-detect.
- Dedicated translation models: **Hunyuan MT 1.5** (Tencent) and **TranslateGemma** (Google).
- Dictionary mode (parts of speech, senses, examples), tone presets, retry / more literal / more natural.
- No cloud APIs, no accounts, nothing leaves your machine.

## Contents

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Models](#models)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Project layout](#project-layout)

## Requirements

| | Ollama backend (recommended) | Hugging Face backend (optional) |
|---|---|---|
| Python | 3.13+ | 3.13+ |
| Python packages | none (standard library only) | PyTorch, Transformers, … (installed into `.venv`) |
| Other software | [Ollama](https://ollama.com/download) | — |
| Accelerator | Apple Silicon, NVIDIA, AMD, or CPU (Ollama handles it) | Apple Silicon (MPS) or NVIDIA (CUDA); CPU works but is slow |
| Disk | ~26 GB for the three default models | ~36 GB for the three default models |
| Memory | 8 GB for translation only; 16 GB+ to also run `gpt-oss` for dictionary mode | 16 GB for the 7B translator; 32 GB+ to keep it and Gemma 4 loaded together |

You can set up either backend, or both. The Ollama backend is simpler, faster (4-bit models), and needs no
Python packages, so start there.

## Setup

Commands below are for **macOS / Linux**. On **Windows** (PowerShell), use `py` instead of `python3`
and `.venv\Scripts\python` instead of `.venv/bin/python` (same for `.venv\Scripts\hf` and `.venv\Scripts\pip`).
Run every command from the repository folder: the scripts read `config.toml` from the current directory.

### 1. Get the code

```bash
git clone https://github.com/xyluo2028/translator.git
cd translator
python3 --version   # must be 3.13 or newer
```

### 2. Ollama backend

1. Install Ollama from https://ollama.com/download and make sure it is running
   (the desktop app starts it automatically; otherwise run `ollama serve`).
2. Pull the models used by the default `config.toml`:

   ```bash
   ollama pull hf.co/tencent/HY-MT1.5-7B-GGUF:Q4_K_M   # Hunyuan MT 1.5 7B, default translator (4.6 GB)
   ollama pull translategemma:12b                      # TranslateGemma 12B, alternative translator (8.1 GB)
   ollama pull gpt-oss:latest                          # general model for dictionary mode and tone (13 GB)
   ```

   Only the first one is required for translating. Skip `gpt-oss` if you don't need dictionary mode or tone,
   or point `dictionary_model` in `config.toml` at another general chat model you already have.
3. Check it works:

   ```bash
   python3 translate.py "Hello, how are you?" --to ZH
   ```

   You should see a Chinese translation followed by `Detected source: EN`.

### 3. Hugging Face backend (optional)

1. Create a virtual environment and install the optional dependencies:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e '.[transformers]'
   ```

   **NVIDIA GPU on Linux/Windows:** first install the CUDA build of PyTorch into the venv with the command that
   https://pytorch.org/get-started/locally/ gives you (pick Pip + your CUDA version; it installs `torch` and
   `torchvision`), then run the `pip install -e` line above. On Apple Silicon the default install already uses
   the GPU (MPS).
2. Download model weights (stored in `~/.cache/huggingface`; set `HF_HOME` to move it):

   ```bash
   .venv/bin/hf download tencent/HY-MT1.5-7B      # default HF translator (16 GB)
   .venv/bin/hf download tencent/HY-MT1.5-1.8B    # small, fast translator (4 GB)
   .venv/bin/hf download google/gemma-4-E4B-it    # general model for dictionary mode and tone (16 GB)
   ```

3. *(Optional)* TranslateGemma on Hugging Face is **gated**. Open
   https://huggingface.co/google/translategemma-4b-it while signed in, accept the license, then:

   ```bash
   .venv/bin/hf auth login                        # paste an access token from huggingface.co/settings/tokens
   .venv/bin/hf download google/translategemma-4b-it   # 8.6 GB
   ```

4. Check it works:

   ```bash
   .venv/bin/python translate.py "Hello, how are you?" --to ZH --provider transformers
   ```

   The first run takes 10–20 s while the model loads.

### 4. Start the web UI

```bash
.venv/bin/python webui.py --open   # if you set up the Hugging Face backend
python3 webui.py --open            # Ollama only
```

This opens http://127.0.0.1:8765. Stop it with Ctrl+C.

## Usage

### Web UI

- **Model** picker (top right) lists every Ollama model you have pulled and the Hugging Face models from
  `config.toml`, grouped by backend. Models that aren't downloaded yet are shown but disabled.
- Choose source (or *Detect language*) and target languages; ⇄ swaps them and moves the result into the input.
- **Translate** / **Dictionary** switch modes. Press ⌘+Enter (macOS) or Ctrl+Enter to run.
- **Retry** samples a new translation. **Tone**, **More literal** and **More natural** need a general model
  (e.g. `gpt-oss`, Gemma 4) and are greyed out for translation-only models.
- **History** keeps your last 30 translations in the browser; click one to restore it.
- Options: `--port 9000`, `--config other.toml`. The server listens on 127.0.0.1 only; `--host 0.0.0.0`
  exposes it to your network with no authentication, so only do that on a network you trust.

### Command line

```bash
python3 translate.py "hello" --to ZH                                   # default model from config.toml
python3 translate.py "How are you?" --to JA --model translategemma:12b # pick a model
python3 translate.py "打ち合わせ" --mode dictionary --to EN              # dictionary lookup
python3 translate.py "How are you?" --to JA --tone polite --model gpt-oss:latest
python3 translate.py "Hello" --from EN --to KO --json --pretty         # structured JSON output
echo "Bonjour tout le monde" | python3 translate.py --to EN             # read from stdin
.venv/bin/python translate.py "hello" --to JA --provider transformers --model tencent/HY-MT1.5-1.8B
```

| Option | Meaning |
|---|---|
| `--to CODE` / `--from CODE` | Target / source language (`EN`, `ZH`, `ZH-TW`, `JA`, `KO`, `FR`, `DE`, `ES`, …; source defaults to `auto`) |
| `--provider ollama\|transformers` | Backend (default: `[provider].name` in `config.toml`) |
| `--model NAME` | Model for this run (Ollama tag or Hugging Face repo id) |
| `--mode translate\|dictionary` | Translation or dictionary lookup |
| `--tone NAME`, `--tone-instructions TEXT` | Style presets (`casual`, `formal`, `polite`, `spoken`, `business`); general models only |
| `--rerun retry\|more_literal\|more_natural` | Regenerate; translation-only models support `retry` only |
| `--json`, `--pretty` | Print the result as JSON |
| `--debug` | On errors, show the raw model output and full traceback |

### Benchmark

`benchmarks/translation_cases.json` holds hand-picked EN/ZH/JA cases; see [benchmarks/README.md](benchmarks/README.md).

```bash
python3 benchmark.py --provider ollama --model translategemma:12b --limit 10
```

## Models

| Backend | Model | Kind | Download | Notes |
|---|---|---|---|---|
| Ollama | `hf.co/tencent/HY-MT1.5-7B-GGUF:Q4_K_M` | translation | 4.6 GB | Default. WMT25-winning model, 33 languages, ~0.5–3 s per sentence |
| Ollama | `translategemma:12b` | translation | 8.1 GB | Google, 55 languages. `translategemma:4b` (3.3 GB) for smaller machines |
| Ollama | `gpt-oss:latest` | general | 13 GB | Dictionary mode, tone, notes, alternatives |
| Hugging Face | `tencent/HY-MT1.5-7B` | translation | 16 GB | Default HF translator (full precision) |
| Hugging Face | `tencent/HY-MT1.5-1.8B` | translation | 4 GB | Nearly as good as 7B, much faster |
| Hugging Face | `google/translategemma-4b-it` | translation | 8.6 GB | Gated: accept license + `hf auth login` |
| Hugging Face | `google/gemma-4-E4B-it` | general | 16 GB | Dictionary mode and tone on the HF backend |

How the app treats them:

- **Translation-only models** (names containing `HY-MT`, `Hunyuan-MT` or `translategemma`) get their official
  prompt format and return only the translation: fast and accurate, but no tone, notes or alternatives.
- **General chat models** (anything else, e.g. `gpt-oss`, `qwen3`, Gemma 4) get a JSON-schema prompt and support
  every feature. Any Ollama chat model you pull shows up in the web UI automatically.
- **Dictionary mode** always runs on a general model. If a translation-only model is selected, the app uses
  `dictionary_model` from `config.toml` for that backend.
- **Auto-detect** with translation-only models guesses from the script: kana → JA, hangul → KO, han → ZH.
  Latin-script text is treated as English and shown as no detection. Japanese written only in kanji is guessed
  as Chinese, and TranslateGemma is told French/German/Spanish text is English, so choose the source language
  explicitly in those cases.

## Configuration

Everything lives in [`config.toml`](config.toml); use `--config path.toml` to load a different file.

```toml
[provider]
name = "ollama"                 # default backend for the CLI: "ollama" or "transformers"

[ollama]
host = "http://localhost:11434"
model = "hf.co/tencent/HY-MT1.5-7B-GGUF:Q4_K_M"   # default translation model
dictionary_model = "gpt-oss:latest"               # general model for dictionary mode + tone

[transformers]
model = "tencent/HY-MT1.5-7B"                    # default translation model
dictionary_model = "google/gemma-4-E4B-it"        # general model for dictionary mode + tone
models = ["tencent/HY-MT1.5-1.8B", "google/translategemma-4b-it"]   # extra models shown in the web UI
device_map = "auto"             # "auto", "mps", "cuda", or "cpu"
dtype = "auto"                  # "auto", "bfloat16", "float16", "float32"
max_new_tokens = 512
enable_thinking = false
trust_remote_code = false

[defaults]
source_lang = "auto"
target_lang = "ZH"
tone = "neutral"
explain_lang = "EN"             # language for notes and explanations
temperature = 0.2
```

## Troubleshooting

| Message | Fix |
|---|---|
| `Failed to reach Ollama at http://localhost:11434 (is ollama serve running?)` | Start the Ollama app or run `ollama serve`. If Ollama runs elsewhere, set `[ollama].host`. |
| `Ollama HTTP 404: model '…' not found` | Pull it: the error prints the exact `ollama pull …` command. Or change the model in `config.toml`. |
| `Transformers backend requires the optional dependencies` | Run with `.venv/bin/python`, not `python3`, or redo [step 3](#3-hugging-face-backend-optional). |
| Hugging Face models show `(run with .venv/bin/python)` in the web UI | Same: start the web UI with `.venv/bin/python webui.py`. |
| Hugging Face models show `(not downloaded)` | `.venv/bin/hf download <repo id>`. |
| `This model is gated: accept its license …` | Accept the license on the model's Hugging Face page, then `.venv/bin/hf auth login`. |
| `requires the PIL library` or `No module named 'torchvision'` | Your venv predates these dependencies: `.venv/bin/pip install -e '.[transformers]'`. |
| `Some parameters are on the meta device because they were offloaded to the disk` | Not enough free memory for the Hugging Face model, so it runs very slowly. Close other models (e.g. a running web UI, Ollama models via `ollama stop <model>`) or use `tencent/HY-MT1.5-1.8B`. |
| First translation is slow | Normal: the model is loading (Ollama ~5–20 s, Hugging Face ~10–30 s). Later requests are fast. |
| `Address already in use` when starting the web UI | Another copy is running; stop it or use `--port 8766`. |
| Translation looks wrong with *Detect language* | Pick the source language explicitly (see auto-detect notes under [Models](#models)). |
| `Warning: You are sending unauthenticated requests to the HF Hub` | Harmless. Logging in (`hf auth login`) or setting `HF_HUB_OFFLINE=1` after downloading silences it. |

## Project layout

```
translate.py              CLI entry point
webui.py                  local web server (standard library only)
benchmark.py              runs benchmarks/translation_cases.json through the CLI
config.toml               backends, models, defaults
translator_app/
  core.py                 routing: picks model + prompt style, parses results
  prompting.py            JSON-schema prompts, HY-MT / TranslateGemma templates, language detection
  ollama.py               Ollama HTTP client
  hf_transformers.py      Hugging Face model loading, caching, generation
  config.py, models.py    config loading, request/result dataclasses
  web/index.html          the web UI (single file, no build step)
```

## Design notes (original plan)

The planning notes this project started from. Setup and usage above reflect the current code.

### Goals

- **Fast translation** between languages with optional **auto-detect**.
- **Tone / style presets** (casual, formal, polite, spoken, business, etc).
- **Dictionary mode** for single words/phrases (multiple senses, POS, examples).
- **Refresh / rerun** to regenerate (optionally with “more literal / more natural”).
- **Provider-agnostic**: local-first with Ollama or Hugging Face Transformers.

### Non-goals (for MVP)

- Perfect human-level translation for every niche domain.
- Full offline bilingual dictionary database (start LLM-based; add later if needed).
- Real-time collaboration / multi-user permissions.

### Product Draft (UX)

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

### Blueprint (Architecture)

Keep the system modular so the UI is a thin client and the “translation engine” can be reused by CLI/API.

#### Components

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
  - `transformers` for local Hugging Face models such as Gemma 4
  - `openai` / `anthropic` / others (API-key based), optional
  - common interface: `generate(prompt, params) -> text`

- **API server (optional but recommended)**
  - endpoints: `/translate`, `/detect`, `/history`
  - handles auth/storage on behalf of UI

- **UI**
  - web app (local server) or desktop wrapper (later)

#### Data flow (Translate)

1. UI sends `TranslateRequest`
2. Server/core normalizes (trim, max length, etc)
3. If source is `Auto`, detect source language
4. Build prompt from: mode + languages + tone + extra instructions + output schema
5. Call provider (Ollama/API)
6. Parse model output into a structured `TranslateResult`
7. Persist (optional): history + cache
8. Return result to UI

### Data contracts (suggested)

#### TranslateRequest

- `text`: string
- `source_lang`: string (`"auto"` allowed)
- `target_lang`: string
- `mode`: `"translate" | "dictionary"`
- `tone`: preset name (e.g. `"casual"`) or `"custom"`
- `tone_instructions`: optional string
- `explain_lang`: language for explanations/notes (e.g. `"EN"`)
- `rerun`: optional object to guide regeneration (e.g. `{ "style": "more_literal" }`)

#### TranslateResult

- `translation`: string
- `alternatives`: string[] (optional)
- `notes`: string (optional)
- `detected_source_lang`: string (when source is auto)
- `provider`: string, `model`: string, `latency_ms`: number (optional)

#### DictionaryResult (when `mode="dictionary"`)

- `term`: string
- `entries`: array of:
  - `pos`: string (noun/verb/adj/…)
  - `senses`: array of:
    - `meaning`: string
    - `example_source`: string (optional)
    - `example_target`: string (optional)
    - `usage_notes`: string (optional)

### Prompting strategy (practical + parseable)

Prefer **structured output** so the UI can render consistently.

- Use a “system-style” instruction: translate accurately, follow tone, no extra text.
- Require **JSON only** output for the selected schema.
- Add guardrails:
  - keep proper nouns unchanged unless requested
  - preserve punctuation and line breaks when reasonable
  - do not invent facts; if ambiguous, add notes

### Tech choices (recommended to start)

If you want fastest iteration:
- **Python** core + **FastAPI** for local API
- Simple web UI (React/Vite) *or* even start with a CLI and add UI later

If you want one-binary distribution later:
- Core stays provider-agnostic; add a desktop wrapper (e.g. Tauri/Electron) after MVP.

### Doable milestone plan

#### M0 — Repo setup (0.5 day)
- Pick language/runtime (recommended: Python)
- Add basic project layout + `config.toml` parsing
- Add a single “hello world” command that prints config/provider info

#### M1 — Core translate engine (1–2 days)
- Define request/result schemas (Translate + Dictionary)
- Implement prompt builder for `translate` mode
- Implement provider adapter for **Ollama**
- Add “refresh” by re-calling with new seed or rerun hint

#### M2 — CLI (0.5–1 day)
- `translate "text" --to ZH --from auto --tone casual`
- `dict "term" --to EN`
- Print structured output + a human-readable view

#### M3 — Local API + minimal UI (1–2 days)
- FastAPI endpoints: `/translate`, `/history` (optional)
- Minimal UI: input, language selectors, tone dropdown, output, refresh

#### M4 — Dictionary mode (1 day)
- Prompt + JSON schema for dictionary output
- UI rendering for multiple senses/examples

#### M5 — Tones + presets (0.5–1 day)
- Preset library (casual/formal/polite/spoken/business)
- “Custom instructions” field merged safely into prompt

#### M6 — Quality + polish (1–2 days)
- History + cache (SQLite)
- Basic eval set + regression checks (hand-curated examples)
- Guardrails: max length, profanity filter option, PII redaction option (optional)

### Open questions

- Desktop vs web vs CLI-first?
- Do we need streaming output (token-by-token) for responsiveness?
- Should we support a user glossary/terminology list early (very useful for names/brands)?
