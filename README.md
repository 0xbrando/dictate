<p align="center">
  <img src="assets/banner.png" alt="Dictate" width="500">
</p>

<h3 align="center">Push-to-talk voice dictation that runs entirely on your Mac.<br>No cloud. No API keys. No subscriptions.</h3>

<p align="center">
  <a href="https://pypi.org/project/dictate-mlx/"><img src="https://img.shields.io/pypi/v/dictate-mlx?color=blue&label=pip" alt="PyPI"></a>
  <a href="https://github.com/0xbrando/dictate/blob/main/LICENSE"><img src="https://img.shields.io/github/license/0xbrando/dictate" alt="License"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-black?logo=apple" alt="Platform">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/tests-1001%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-97%25-brightgreen" alt="Coverage">
</p>

<p align="center">
  <b>Hold a key → Speak → Release → Clean text appears wherever your cursor is.</b>
</p>

<!-- TODO: Replace with actual demo GIF once recorded -->
<!-- <p align="center"><img src="assets/demo.gif" alt="Dictate demo" width="600"></p> -->

---

## Why Dictate?

| | Dictate | SuperWhisper | Wispr Flow | VoiceInk | macOS Dictation |
|---|:---:|:---:|:---:|:---:|:---:|
| **Price** | **Free** | $8.49/mo | $12/mo | $39.99 | Free |
| **Open source** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **100% local** | ✅ | ✅ | ❌ (cloud) | ✅ | Partial |
| **LLM cleanup** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Translation** | ✅ 12 langs | ❌ | ❌ | ❌ | ❌ |
| **Writing styles** | ✅ 8 modes | ❌ | ✅ | ❌ | ❌ |
| **Push-to-talk** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Personal dictionary** | ✅ | ❌ | ✅ | ❌ | ✅ |

Your M-series Mac has more than enough compute to do this locally. Why pay for it?

## Install

```bash
pip install dictate-mlx
dictate
```

That's it. Dictate launches in the background and appears in your menu bar. Close the terminal — it keeps running.

<img src="assets/menubar-icon.png" alt="Dictate in the menu bar">

macOS will prompt for **Accessibility** and **Microphone** permissions on first run. Models download automatically (~2-4GB, cached in `~/.cache/huggingface/`).

<details>
<summary><b>Install from source</b></summary>

```bash
git clone https://github.com/0xbrando/dictate.git
cd dictate
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
dictate
```
</details>

### Requirements

- macOS with Apple Silicon (any M-series chip)
- Python 3.11+
- ~4GB RAM minimum, ~6GB recommended

## Features

### 🎙️ Push-to-Talk

Hold a key, speak, release. Text appears wherever your cursor is.

| Action | Key |
|--------|-----|
| Record | Hold Left Control |
| Lock recording (hands-free) | Press Space while holding PTT |
| Stop locked recording | Press PTT again |

The PTT key is configurable: Left Control, Right Control, Right Command, or either Option key.

### 🧠 LLM Text Cleanup

**The thing that sets Dictate apart.** Most dictation tools give you raw transcription. Dictate pipes through a local LLM that fixes grammar, adds punctuation, and formats properly.

Short phrases (≤15 words) skip cleanup for instant speed. Longer dictation gets the full treatment.

### 🗣️ Two STT Engines

Both included. Switch anytime from the menu bar.

| Engine | Speed | Languages | Notes |
|--------|-------|-----------|-------|
| **Parakeet TDT 0.6B** | ~50ms | English | Default — 4-8x faster than Whisper |
| **Whisper Large V3 Turbo** | ~300ms | 99+ | For multilingual or non-English input |

### ✍️ Writing Styles

<img src="assets/writing-style.png" alt="Writing styles menu" width="400">

| Style | What it does |
|-------|-------------|
| **Clean Up** | Fixes punctuation and capitalization — keeps your words |
| **Formal** | Rewrites in a professional tone |
| **Bullet Points** | Distills dictation into concise key points |
| **Email** | Formats as a professional email with greeting and sign-off |
| **Slack / Chat** | Casual, concise, chat-friendly messages |
| **Technical** | Precise technical documentation style |
| **Tweet** | Condenses to under 280 characters |
| **Raw** | No LLM processing — exact transcription output |

8 writing styles — more than any macOS dictation tool.

### 🌐 Real-Time Translation

Speak in one language, get output in another. 12 languages supported: English, Spanish, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Russian, Arabic, Hindi.

### ⚡ Quality Presets

<img src="assets/quality.png" alt="Quality presets menu" width="400">

| Preset | Speed | RAM | Best for |
|--------|-------|-----|----------|
| API Server | varies | 0 | Use your own LLM server (LM Studio, Ollama, etc.) |
| Speedy (1.5B) | ~120ms | 1.0GB | Quick fixes, any M-chip |
| Fast (3B) | ~250ms | 1.8GB | Everyday use |
| Balanced (7B) | ~350ms | 4.2GB | Longer dictation, formal rewrites |
| Quality (14B) | ~500ms | 8.8GB | Best accuracy |

Smart routing auto-routes by message length: short phrases → fast local model, longer dictation → larger model or API server.

Times on M3 Ultra. The app picks the best default for your chip.

## Menu Bar

<p align="center">
  <img src="assets/menubar.png" alt="Main menu" width="250">
  <img src="assets/advanced.png" alt="Advanced settings" width="250">
</p>

Everything accessible from the waveform icon:

- **Writing Style** — Clean Up, Formal, Bullet Points, Email, Slack, Technical, Tweet, Raw
- **Quality** — model size (shows only downloaded models)
- **Input Device** — select microphone
- **Recent** — last 10 transcriptions, click to re-paste
- **STT Engine** — Whisper or Parakeet
- **PTT Key** — choose your push-to-talk modifier
- **Languages** — input and output language
- **Sounds** — 6 notification tones or silent
- **Personal Dictionary** — names, brands, technical terms always spelled correctly
- **Launch at Login** — auto-start on boot

## API Server

If you run a local LLM server, Dictate can use it instead of loading its own model — zero additional RAM:

```bash
DICTATE_LLM_BACKEND=api DICTATE_LLM_API_URL=http://localhost:8005/v1/chat/completions dictate
```

Works with any OpenAI-compatible server: [vllm-mlx](https://github.com/vllm-project/vllm-mlx), [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com).

The **Smart** preset auto-routes by length: short phrases → fast local model (~120ms), longer dictation → your API server.

## Environment Variables

<details>
<summary><b>All environment variables</b></summary>

| Variable | Description | Default |
|----------|-------------|---------|
| `DICTATE_AUDIO_DEVICE` | Microphone device index | System default |
| `DICTATE_OUTPUT_MODE` | `type` or `clipboard` | `type` |
| `DICTATE_INPUT_LANGUAGE` | `auto`, `en`, `ja`, `ko`, etc. | `auto` |
| `DICTATE_OUTPUT_LANGUAGE` | Translation target (`auto` = same) | `auto` |
| `DICTATE_LLM_CLEANUP` | Enable LLM text cleanup | `true` |
| `DICTATE_LLM_MODEL` | `qwen-1.5b`, `qwen`, `qwen-7b`, `qwen-14b` | `qwen` |
| `DICTATE_LLM_BACKEND` | `local` or `api` | `local` |
| `DICTATE_LLM_API_URL` | OpenAI-compatible endpoint | `http://localhost:8005/v1/chat/completions` |
| `DICTATE_ALLOW_REMOTE_API` | Allow non-localhost API URLs | unset |

</details>

## Agent Integration

Dictate works well as a voice input layer for AI assistants and agent frameworks. If you're building with tools like Claude Code, OpenClaw, or similar — Dictate gives your setup a local, private voice interface with zero cloud dependency.

## CLI Commands

```bash
dictate              # Launch in menu bar (backgrounds automatically)
dictate config       # View all preferences
dictate config set writing_style formal
dictate config set quality speedy
dictate config set ptt_key cmd_r
dictate config set stt whisper
dictate config reset # Reset to defaults
dictate stats        # Show usage statistics
dictate status       # System info and model status
dictate doctor       # Run diagnostic checks (troubleshooting)
dictate devices      # List audio input devices
dictate update       # Update to latest version
dictate -f           # Run in foreground (debug)
dictate -V           # Show version
```

### Config Keys

| Key | Values |
|-----|--------|
| `writing_style` | clean, formal, bullets, email, slack, technical, tweet, raw |
| `quality` | api, speedy, fast, balanced, quality |
| `stt` | parakeet, whisper |
| `input_language` | auto, en, ja, de, fr, es, ... |
| `output_language` | auto, en, ja, de, fr, es, ... |
| `ptt_key` | ctrl_l, ctrl_r, cmd_r, alt_l, alt_r |
| `llm_cleanup` | on, off |
| `sound` | soft_pop, chime, warm, click, marimba, simple |
| `llm_endpoint` | host:port (for API backend) |

## Shell Completions

Tab completions for bash and zsh:

```bash
# Bash — add to ~/.bashrc
source /path/to/dictate/completions/dictate.bash

# Zsh — copy to fpath dir, then reload
cp completions/dictate.zsh ~/.zsh/completions/_dictate
autoload -Uz compinit && compinit
```

Completes commands, config keys, and all valid values.

## Debugging

```bash
# Run in foreground with logs
dictate --foreground

# Check background logs
tail -f ~/Library/Logs/Dictate/dictate.log
```

## Security

- All processing is local. Audio and text never leave your machine.
- LLM endpoints restricted to localhost by default (`DICTATE_ALLOW_REMOTE_API=1` to override).
- Preferences stored with `0o600` permissions (owner-only).
- No API keys, tokens, or accounts required.
- 994 tests, 97% code coverage.

## Contributing

Issues and PRs welcome. Run the test suite before submitting:

```bash
python -m pytest tests/ -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — See [LICENSES.md](LICENSES.md) for dependency licenses.
