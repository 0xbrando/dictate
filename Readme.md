# Dictate

Push-to-talk voice dictation that lives in your macOS menu bar. 100% local — all processing runs on-device using Apple Silicon MLX models. No cloud, no API keys, no subscriptions.

## Features

- **Menu bar app** with reactive waveform icon that follows your voice
- **Push-to-talk**: Hold a key to record, release to transcribe
- **Configurable PTT key**: Left Control, Right Control, Right Command, or either Option key
- **Lock recording**: Press Space while holding PTT for hands-free dictation
- **Auto-type**: Pastes directly into the focused window
- **Smart routing**: Short messages use a fast local model, long messages route to your API server
- **LLM cleanup**: Fixes grammar and punctuation using local AI models
- **Writing styles**: Clean Up, Formal, or Bullet Points mode
- **Personal dictionary**: Teach it names, brands, and technical terms it should always spell correctly
- **Translation**: Transcribe in one language, output in another (12 languages)
- **Dual STT engines**: Whisper (99+ languages) or Parakeet (4-8x faster, English)
- **Quality presets**: Smart, Speedy (1.5B), Fast (3B), Balanced (7B), Quality (14B)
- **Sound presets**: 6 synthesized tones (Soft Pop, Chime, Warm, Click, Marimba, Simple) or silent
- **Pause/Resume**: Toggle dictation on and off without quitting
- **Launch at Login**: Auto-start when you turn on your Mac
- **Recent transcriptions**: Last 10 items, click to re-paste
- **Hardware auto-detection**: Picks the best quality preset for your chip on first launch
- **Singleton lock**: Prevents duplicate instances from running simultaneously
- **100% private**: Everything runs locally. No data ever leaves your machine.

All settings persist between sessions.

## Requirements

- macOS with Apple Silicon (any M-series chip)
- Python 3.11+
- ~4GB RAM minimum (Speedy preset), ~6GB recommended (Balanced)

## Installation

```bash
git clone https://github.com/0xbrando/dictate.git
cd dictate

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Models download automatically on first run (~2GB for Whisper, ~1-8GB for LLM depending on quality preset).

### Optional: Parakeet STT (faster, English-only)

```bash
pip install parakeet-mlx
```

Parakeet is 4-8x faster than Whisper but only supports English. Select it from the menu bar after installing.

## Usage

```bash
source .venv/bin/activate
python -m dictate
```

| Action | Default Key |
|--------|-------------|
| Record | Hold Left Control |
| Lock Recording | Press Space while holding PTT |
| Stop Locked Recording | Press PTT again |

The PTT key is configurable from the menu bar. macOS will prompt for Accessibility and Microphone permissions on first run.

### First Launch

On first launch, Dictate downloads the Whisper STT model (~2GB) and an LLM cleanup model (~1-8GB depending on your chip). This happens in the background — you'll see progress in the menu bar status:

1. Click the waveform icon in your menu bar
2. The top status line shows download/loading progress (e.g. "Downloading Whisper (~2GB)...")
3. When you see **"Ready"**, you're good to go

Downloads are cached in `~/.cache/huggingface/` so this only happens once. On a fast connection it takes 1-2 minutes; on slower connections it may take longer. The app auto-detects your chip and picks the best model size — Ultra/Max chips get the 3B model, everything else gets the faster 1.5B.

### Menu Bar Options

All settings are accessible from the menu bar icon:

- **Pause/Resume Dictation** — stop listening without quitting
- **Microphone** — select input device
- **Push-to-Talk Key** — choose which modifier key triggers recording
- **Quality** — choose model size (only shows downloaded models)
- **Sounds** — pick recording start/stop tones (with preview)
- **Writing Style** — Clean Up, Formal, or Bullet Points
- **Input/Output Language** — transcription and translation settings
- **LLM Cleanup** — toggle AI text cleanup on/off
- **Recent** — click any recent transcription to re-paste it
- **Launch at Login** — auto-start on boot

## Writing Styles

| Style | What it does |
|-------|-------------|
| **Clean Up** | Fixes punctuation and capitalization — keeps your words |
| **Formal** | Rewrites in a professional tone |
| **Bullet Points** | Distills your dictation into concise key points |

## Quality Presets

| Preset | Speed | RAM | Best for |
|--------|-------|-----|----------|
| Smart | ~250ms | 0 | Auto-routes: fast local for short, API server for long |
| Speedy — 1.5B | ~120ms | 1GB | Quick fixes, great for any chip |
| Fast — 3B | ~250ms | 2GB | Quick cleanup, everyday use |
| Balanced — 7B | ~350ms | 5GB | Longer dictation, formal rewriting |
| Quality — 14B | ~500ms | 9GB | Best accuracy for bullet points and rewrites |

All times measured on Mac Studio M3 Ultra. Whisper transcription adds ~300ms (Parakeet adds ~50ms).

The Quality menu only shows models you've already downloaded — no clutter from presets you can't use yet. To add a model, run it once from the command line:

```bash
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-7B-Instruct-4bit')"
```

### Smart Routing (API Mode)

When using the Smart preset with a local API server, Dictate automatically routes:
- **Short messages** (15 words or fewer) to the fastest cached local model (~120ms)
- **Long messages** (16+ words) to your API server for higher quality

This gives you the speed of a small model for quick dictations and the quality of a large model for longer text — without any manual switching.

### API Server Setup

If you run a local LLM server (vllm-mlx, LM Studio, Ollama, etc.), Dictate can use it instead of loading a bundled model — zero additional RAM. Point it at any OpenAI-compatible endpoint:

```bash
DICTATE_LLM_BACKEND=api DICTATE_LLM_API_URL=http://localhost:8005/v1/chat/completions python -m dictate
```

**Recommended models for dictation cleanup:**

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| Qwen2.5-3B-Instruct-4bit | 2GB | ~250ms | Best speed/quality ratio |
| Qwen2.5-7B-Instruct-4bit | 5GB | ~350ms | Better for formal rewriting and bullet points |
| Qwen3-Coder-Next (80B MoE) | 50GB | ~650ms | Great if you already run it for other tasks |

**Serving options:**
- [vllm-mlx](https://github.com/vllm-project/vllm-mlx) — fastest for Apple Silicon, OpenAI-compatible out of the box
- [LM Studio](https://lmstudio.ai) — GUI, easy model management, local server built in
- [Ollama](https://ollama.com) — `ollama run qwen2.5:3b` then point Dictate at `http://localhost:11434/v1/chat/completions`

## STT Engines

| Engine | Speed | Languages | Notes |
|--------|-------|-----------|-------|
| **Whisper Large V3 Turbo** | ~300ms | 99+ | Default, most versatile |
| **Parakeet TDT 0.6B** | ~50ms | English | `pip install parakeet-mlx`, 4-8x faster |

Switch engines from the menu bar (when Parakeet is installed).

## How It Works

```
Mic → VAD → STT (Whisper or Parakeet) → Smart Skip / LLM Cleanup → Auto-paste
```

1. **Push-to-talk** captures audio via the microphone
2. **VAD** (voice activity detection) segments speech from silence
3. **STT** transcribes locally — Whisper Large V3 Turbo or Parakeet TDT
4. **Smart skip** detects clean short utterances and skips the LLM entirely
5. **LLM** cleans up, rewrites, or converts to bullet points (with smart routing in API mode)
6. **Auto-paste** puts the result into the focused window

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
| `DICTATE_ALLOW_REMOTE_API` | Set to `1` to allow non-localhost API URLs | unset |

## License

MIT — See [LICENSES.md](LICENSES.md) for dependency licenses.
