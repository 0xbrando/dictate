# Changelog

All notable changes to Dictate are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **5 new writing styles** — Email, Slack/Chat, Technical, Tweet, Raw (8 total)
- **`dictate status` command** — shows system info, models, preferences, running state
- **Raw mode** — bypasses LLM cleanup entirely for exact transcription output

### Fixed
- Test regex mismatch with new error message from robustness merge
- README test badge updated (871 tests)

## [2.4.1] - 2026-02-10

### Fixed
- Add space between consecutive dictation pastes — no more words running together

## [2.4.0] - 2026-02-09

### Added
- Colored launch banner with usage guide, tips, and commands on first run
- Strip `<think>` tags from reasoning model output — clean text from any LLM
- SEO keywords in pyproject.toml for better discoverability

### Changed
- Polished screenshots with rounded corners and "100% local" tagline
- Menu bar icon screenshot added to install section
- Docs overhaul

## [2.3.1] - 2026-02-07

### Fixed
- Replace `os.fork()` with `subprocess.Popen` for daemonization — fixes crash on macOS with ObjC runtime
- Sync `__version__` and pyproject.toml version strings
- Security: fix 6 medium issues from audit (endpoint validation, plist hardening)

## [2.3.0] - 2026-02-06

### Added
- Auto update check on launch — notifies when a new version is available
- `dictate update` command for one-step upgrades

### Fixed
- Status item no longer greyed out after model reload
- Reorder Advanced menu — toggles at top, grouped by category

## [2.2.1] - 2026-02-05

### Fixed
- macOS platform guard — prevents confusing install on Linux/Windows
- Restore `dictate` command entry point after package rename
- Smaller status dots using Unicode symbols

## [2.2.0] - 2026-02-04

### Added
- Manage Models menu — view cache size, delete models, open cache folder
- Model download progress now visible in Quality menu

## [2.1.0] - 2026-02-03

### Added
- Colored status dots: green (ready), red (error/paused), yellow (loading)
- Simple top-level menu layout with Advanced nested submenu
- Parakeet set as default STT engine for speed

### Changed
- Package renamed to `dictate-mlx` for PyPI
- README updated with `pip install` instructions and new menu layout

### Fixed
- Hardened error handling across 5 modules
- Daemonize on launch — close terminal, Dictate keeps running

## [2.0.0] - 2026-01-28

### Added
- Push-to-talk voice dictation — hold key, speak, release, text appears
- 100% local processing using Apple Silicon MLX
- Dual STT engines: Whisper Large V3 Turbo (99+ languages) and Parakeet TDT 0.6B (English, 4-8x faster)
- LLM cleanup with smart skip for clean short phrases
- Quality presets: Draft (fast), Balanced, Quality, Max
- Writing styles: Clean, Formal, Bullets
- Personal dictionary for domain-specific terms
- Customizable push-to-talk key (Left/Right Control, Right Command, Option)
- Lock recording mode — press Space while holding PTT for hands-free
- Auto-paste into active window via clipboard
- Hardware auto-detection for optimal defaults on M1/M2/M3/M4
- LLM endpoint discovery for local models (Ollama, LM Studio, vLLM)
- Singleton lock — prevents duplicate instances
- Sound feedback on record start/stop
- Pause/resume from menu bar
- Launch at login option
- Recent transcriptions in menu bar
- 274 tests

### Security
- Endpoint validation and plist generation hardening
- Model repository allowlist (mlx-community/ prefix only)

[2.4.1]: https://github.com/0xbrando/dictate/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/0xbrando/dictate/compare/v2.3.1...v2.4.0
[2.3.1]: https://github.com/0xbrando/dictate/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/0xbrando/dictate/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/0xbrando/dictate/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/0xbrando/dictate/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/0xbrando/dictate/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/0xbrando/dictate/releases/tag/v2.0.0
