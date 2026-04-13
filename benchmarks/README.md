# Benchmark Suite

This directory contains hand-curated benchmark cases for translation quality checks.

Current scope:

- Mode: `translate`
- Languages: `EN`, `ZH`, `JA`
- Directions covered:
  - `EN -> ZH`
  - `ZH -> EN`
  - `EN -> JA`
  - `JA -> EN`
  - `ZH -> JA`
  - `JA -> ZH`

Files:

- `translation_cases.json`: benchmark inputs, references, and evaluation focus notes
- `../benchmark.py`: runner that executes the suite through `translate.py` and writes JSONL results

Suggested scoring rubric:

- `accuracy` (1-5): meaning preservation, omissions, additions, mistranslations
- `fluency` (1-5): natural output in the target language
- `register` (1-5): tone and politeness match the source
- `format` (1-5): preservation of line breaks, bullets, punctuation, and numbers
- `instruction_following` (pass/fail): output is only the requested translation

Suggested usage:

1. Use `request_source_lang` as the CLI `--from` value.
2. Use `target_lang` as the CLI `--to` value.
3. Compare model output against `reference_primary` and `reference_alternatives`.
4. Use `evaluation_focus` to judge acceptable variation.

Example:

```bash
python3 translate.py "How are you?" --from auto --to JA --provider transformers
```

Dry-run the suite:

```bash
python3 benchmark.py --provider transformers --dry-run
```

Run the full suite and save JSONL results:

```bash
python3 benchmark.py --provider transformers
```

Run only one direction:

```bash
python3 benchmark.py --provider transformers --source-lang EN --target-lang JA
```
