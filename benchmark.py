#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise SystemExit(f"Invalid benchmark file: {path}")
    return data, cases


def _filter_cases(
    cases: list[dict[str, Any]],
    *,
    case_ids: set[str] | None,
    source_lang: str | None,
    target_lang: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for case in cases:
        if case_ids and case.get("id") not in case_ids:
            continue
        if source_lang and case.get("source_lang") != source_lang:
            continue
        if target_lang and case.get("target_lang") != target_lang:
            continue
        filtered.append(case)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def _default_output_path(results_dir: Path, suite_name: str, provider: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{suite_name}_{provider}_{stamp}.jsonl"


def _run_case(
    repo_root: Path,
    case: dict[str, Any],
    *,
    config: str,
    provider: str,
    timeout_s: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "translate.py",
        str(case["source_text"]),
        "--config",
        config,
        "--from",
        str(case["request_source_lang"]),
        "--to",
        str(case["target_lang"]),
        "--provider",
        provider,
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    output_json = None
    if proc.returncode == 0 and stdout:
        try:
            output_json = json.loads(stdout)
        except json.JSONDecodeError:
            output_json = None

    return {
        "id": case["id"],
        "source_lang": case["source_lang"],
        "target_lang": case["target_lang"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "source_text": case["source_text"],
        "reference_primary": case["reference_primary"],
        "reference_alternatives": case.get("reference_alternatives", []),
        "evaluation_focus": case.get("evaluation_focus", []),
        "provider": provider,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "output_json": output_json,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the translation benchmark suite through translate.py.")
    parser.add_argument(
        "--suite",
        default="benchmarks/translation_cases.json",
        help="Path to the benchmark suite JSON file.",
    )
    parser.add_argument("--config", default="config.toml", help="Path to the translator config TOML.")
    parser.add_argument("--provider", choices=("ollama", "transformers"), required=True)
    parser.add_argument("--output", default=None, help="Path to write JSONL results.")
    parser.add_argument("--results-dir", default="benchmarks/results", help="Directory for generated result files.")
    parser.add_argument("--case-id", action="append", default=[], help="Run only the specified case id. Repeatable.")
    parser.add_argument("--source-lang", choices=("EN", "ZH", "JA"), default=None, help="Filter by source language.")
    parser.add_argument("--target-lang", choices=("EN", "ZH", "JA"), default=None, help="Filter by target language.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N filtered cases.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-case timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="List selected cases without executing them.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failing case.")

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    suite_path = (repo_root / args.suite).resolve()
    suite_data, cases = _load_cases(suite_path)
    selected = _filter_cases(
        cases,
        case_ids=set(args.case_id) if args.case_id else None,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        limit=args.limit,
    )

    if not selected:
        raise SystemExit("No benchmark cases matched the requested filters.")

    if args.dry_run:
        for case in selected:
            print(f"{case['id']}\t{case['source_lang']}->{case['target_lang']}\t{case['category']}\t{case['difficulty']}")
        return 0

    results_dir = (repo_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).resolve() if args.output else _default_output_path(
        results_dir,
        str(suite_data.get("suite_name", "benchmark")),
        args.provider,
    )

    failures = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for idx, case in enumerate(selected, start=1):
            print(f"[{idx}/{len(selected)}] {case['id']} {case['source_lang']}->{case['target_lang']}", file=sys.stderr)
            try:
                record = _run_case(
                    repo_root,
                    case,
                    config=args.config,
                    provider=args.provider,
                    timeout_s=args.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                record = {
                    "id": case["id"],
                    "source_lang": case["source_lang"],
                    "target_lang": case["target_lang"],
                    "category": case["category"],
                    "difficulty": case["difficulty"],
                    "source_text": case["source_text"],
                    "reference_primary": case["reference_primary"],
                    "reference_alternatives": case.get("reference_alternatives", []),
                    "evaluation_focus": case.get("evaluation_focus", []),
                    "provider": args.provider,
                    "command": exc.cmd,
                    "returncode": None,
                    "stdout": (exc.stdout or "").strip(),
                    "stderr": (exc.stderr or "").strip(),
                    "output_json": None,
                    "timeout": True,
                }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record.get("returncode") != 0:
                failures += 1
                if args.fail_fast:
                    break

    print(f"results={output_path}")
    print(f"cases={len(selected)}")
    print(f"failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
