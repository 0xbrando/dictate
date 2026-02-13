# Contributing to Dictate

Thanks for your interest in improving Dictate!

## Quick Start

```bash
git clone https://github.com/0xbrando/dictate.git
cd dictate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -q
```

All PRs should maintain the existing test coverage (~97%). New features should include tests.

## Code Style

- Python 3.11+
- Black formatter, Ruff linter
- Type hints encouraged
- Docstrings for public methods

## What to Work On

Check the [issues page](https://github.com/0xbrando/dictate/issues) for open items. Good first contributions:

- **Bug reports** — If something breaks, file an issue with your macOS version and chip
- **STT engine improvements** — Better VAD, faster transcription paths
- **Writing styles** — New cleanup/rewrite modes
- **Language support** — Test and improve non-English transcription
- **Documentation** — Tutorials, examples, translations

## Architecture

```
dictate/
├── main.py          # Entry point, daemon management
├── menubar.py       # Menu bar UI (rumps)
├── transcribe.py    # STT pipeline (Whisper + Parakeet)
├── llm.py           # LLM text cleanup
├── audio.py         # Audio capture, VAD
└── config.py        # Preferences, env vars
```

## PR Guidelines

1. Branch from `main`
2. One concern per PR (don't mix features with bug fixes)
3. Run the full test suite before submitting
4. Describe what changed, why, and how to test it

## macOS-Specific Notes

- The menu bar UI uses [rumps](https://github.com/jaredks/rumps) — mock it for unit tests
- MLX operations require Apple Silicon — tests mock the MLX layer
- Accessibility (keyboard simulation) requires permissions — tests mock this
- `os.fork()` is incompatible with ObjC/AppKit — use `subprocess.Popen` for daemon mode

## Questions?

Open an issue or start a discussion on the repo.
