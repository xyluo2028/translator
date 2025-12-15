#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from translator_app.config import load_config
from translator_app.core import translate_text
from translator_app.models import RerunHint, TranslateRequest


def _read_text_from_stdin() -> str:
    data = sys.stdin.read()
    if not data:
        raise SystemExit("No input provided (pass TEXT arg or pipe via stdin).")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-backed translator (Ollama by default).")
    parser.add_argument("text", nargs="?", help="Text to translate (or omit to read stdin).")

    parser.add_argument("--config", default="config.toml", help="Path to config TOML.")
    parser.add_argument("--mode", choices=("translate", "dictionary"), default=None)
    parser.add_argument("--from", dest="source_lang", default=None, help='Source language (or "auto").')
    parser.add_argument("--to", dest="target_lang", default=None, help="Target language.")

    parser.add_argument("--tone", default=None, help='Tone preset (e.g. "casual", "formal").')
    parser.add_argument("--tone-instructions", default=None, help="Additional style instructions.")
    parser.add_argument("--explain-lang", default=None, help="Language for notes/explanations.")

    parser.add_argument(
        "--rerun",
        choices=("retry", "more_literal", "more_natural"),
        default=None,
        help="Regenerate with a rerun hint.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed to vary results.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override.")

    parser.add_argument("--json", action="store_true", help="Print JSON result only.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--debug", action="store_true", help="Print raw provider response on errors.")

    args = parser.parse_args(argv)

    text = args.text if args.text is not None else _read_text_from_stdin()

    config = load_config(args.config)

    request = TranslateRequest(
        text=text,
        source_lang=args.source_lang or config.defaults.source_lang,
        target_lang=args.target_lang or config.defaults.target_lang,
        mode=args.mode or "translate",
        tone=args.tone or config.defaults.tone,
        tone_instructions=args.tone_instructions,
        explain_lang=args.explain_lang or config.defaults.explain_lang,
        rerun=RerunHint(style=args.rerun) if args.rerun else None,
        seed=args.seed,
        temperature=args.temperature if args.temperature is not None else config.defaults.temperature,
    )

    try:
        result = translate_text(request, config=config)
    except Exception as exc:
        if args.debug and getattr(exc, "raw_response", None):
            print("=== raw_response ===", file=sys.stderr)
            print(exc.raw_response, file=sys.stderr)
            print("=== end raw_response ===", file=sys.stderr)
        raise

    payload = asdict(result)

    if args.json:
        if args.pretty:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return 0

    if request.mode == "dictionary":
        print(f"Term: {payload.get('term')}")
        for entry in payload.get("entries", []):
            pos = entry.get("pos") or "—"
            print(f"\n[{pos}]")
            for idx, sense in enumerate(entry.get("senses", []), start=1):
                print(f"{idx}. {sense.get('meaning')}")
                ex_s = sense.get("example_source")
                ex_t = sense.get("example_target")
                if ex_s or ex_t:
                    print(f"   e.g. {ex_s or ''} -> {ex_t or ''}".rstrip())
                notes = sense.get("usage_notes")
                if notes:
                    print(f"   note: {notes}")
        return 0

    print(payload.get("translation", ""))
    alts = payload.get("alternatives") or []
    if alts:
        print("\nAlternatives:")
        for alt in alts:
            print(f"- {alt}")
    notes = payload.get("notes")
    if notes:
        print(f"\nNotes:\n{notes}")
    detected = payload.get("detected_source_lang")
    if detected:
        print(f"\nDetected source: {detected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

