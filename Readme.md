# Dictate

Push-to-talk voice dictation for macOS. Runs 100% on-device using Apple Silicon MLX models. No cloud, no API keys, no subscriptions.

Hold a key, speak, release — clean text appears wherever your cursor is.

## Quick Start

```bash
git clone https://github.com/0xbrando/dictate.git
cd dictate
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
dictate
```

That's it. Dictate launches in the background and appears in your menu bar. Close the terminal — it keeps running. Quit from the menu bar icon.

macOS will prompt for **Accessibility** and **Microphone** permissions on first run. Models download automatically in the background (~2-4GB total, cached in `~/.cache/huggingface/`).

## Requirements

- macOS with Apple Silicon (any M-series chip)
- Python 3.11+
- ~4GB RAM minimum, ~6GB recommended

## How It Works

```
Hold PTT → Speak → Release → Clean text pasted into active window
```

Under the hood:

1. **Push-to-talk** captures audio via the microphone
2. **VAD** segments speech from silence
3. **STT** transcribes locally (Whisper or Parakeet)
4. **Smart skip** detects clean short phrases and skips cleanup entirely
5. **LLM** fixes grammar, punctuation, and formatting
6. **Auto-paste** puts the result wherever your cursor is

Everything runs locally. Nothing leaves your machine.

## Controls

| Action | Key |
|--------|-----|
| Record | Hold Left Control |
| Lock recording (hands-free) | Press Space while holding PTT |
| Stop locked recording | Press PTT again |

The PTT key is configurable from the menu bar: Left Control, Right Control, Right Command, or either Option key.

## STT Engines

Both engines are included. Switch anytime from the menu bar.

| Engine | Speed | Languages | Notes |
|--------|-------|-----------|-------|
| **Parakeet TDT 0.6B** | ~50ms | English | Default. 4-8x faster than Whisper |
| **Whisper Large V3 Turbo** | ~300ms | 99+ | Best for multilingual or non-English |

Parakeet is the default for speed. Switch to Whisper from the menu bar if you need non-English STT.

## Writing Styles

| Style | What it does |
|-------|-------------|
| **Clean Up** | Fixes punctuation and capitalization — keeps your words |
| **Formal** | Rewrites in a professional tone |
| **Bullet Points** | Distills your dictation into concise key points |
| **Email** | Formats as a professional email with greeting and sign-off |
| **Code Comment** | Converts speech into clean code comments and docstrings |

## Quality Presets

| Preset | Speed | RAM | Best for |
|--------|-------|-----|----------|
| Smart | ~250ms | 0 | Auto-routes: fast local for short, API server for long |
| Speedy (1.5B) | ~120ms | 1GB | Quick fixes, great for any chip |
| Fast (3B) | ~250ms | 2GB | Quick cleanup, everyday use |
| Balanced (7B) | ~350ms | 5GB | Longer dictation, formal rewriting |
| Quality (14B) | ~500ms | 9GB | Best accuracy for bullet points and rewrites |

Times measured on M3 Ultra. The app picks the best default for your chip — Ultra/Max get 3B, everything else gets 1.5B.

The Quality menu only shows models you've downloaded. To add a larger model:

```bash
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-7B-Instruct-4bit')"
```

## Menu Bar Options

All settings are accessible from the waveform icon in your menu bar:

- **Pause/Resume** — stop listening without quitting
- **Microphone** — select input device
- **PTT Key** — choose your push-to-talk modifier
- **STT Engine** — Whisper or Parakeet
- **Quality** — model size (shows only downloaded models)
- **Sounds** — 6 tones or silent
- **Writing Style** — Clean Up, Formal, Bullet Points, Email, or Code Comment
- **Languages** — input and output language (12 languages for translation)
- **LLM Cleanup** — toggle on/off
- **Personal Dictionary** — names, brands, technical terms always spelled correctly
- **Recent** — last 10 transcriptions, click to re-paste
- **Launch at Login** — auto-start on boot
- **Quit** — stop Dictate

## API Server Setup

If you run a local LLM server, Dictate can use it instead of loading its own model — zero additional RAM:

```bash
DICTATE_LLM_BACKEND=api DICTATE_LLM_API_URL=http://localhost:8005/v1/chat/completions dictate
```

Works with any OpenAI-compatible server: [vllm-mlx](https://github.com/vllm-project/vllm-mlx), [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com).

### Smart Routing

The Smart preset auto-routes based on message length:
- **Short** (15 words or fewer) → fast local model (~120ms)
- **Long** (16+ words) → your API server for higher quality

## Environment Variables

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

## Agent Integration

Dictate works well as a voice input layer for AI assistants and agent frameworks. If you're building with tools like Claude Code, OpenClaw, or similar — Dictate gives your setup a local, private voice interface with zero cloud dependency.

Your assistant can help install Dictate by following the Quick Start above. Once running, dictation output goes directly into whatever window has focus — including terminals, chat interfaces, and code editors.

## Debugging

Run in the foreground to see logs:

```bash
dictate --foreground
```

Or check the background log:

```bash
tail -f ~/Library/Logs/Dictate/dictate.log
```

## Security

- All processing is local. Audio and text never leave your machine.
- LLM endpoints are restricted to localhost by default. Set `DICTATE_ALLOW_REMOTE_API=1` to override.
- Preferences stored with `0o600` permissions (owner-only read/write).
- No API keys, tokens, or accounts required.

## License

MIT — See [LICENSES.md](LICENSES.md) for dependency licenses.
