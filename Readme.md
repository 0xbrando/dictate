# Dictate

Push-to-talk voice dictation that lives in your macOS menu bar. 100% local — all processing runs on-device using Apple Silicon MLX models. No cloud, no API keys, no subscriptions.

## Features

- **Menu bar app** with reactive waveform icon that follows your voice
- **Push-to-talk**: Hold Control to record, release to transcribe
- **Lock recording**: Press Space while holding Control for hands-free
- **Auto-type**: Pastes directly into the focused window
- **LLM cleanup**: Fixes grammar and punctuation using local AI models
- **Writing styles**: Clean Up, Formal, or Bullet Points mode
- **Translation**: Transcribe in one language, output in another (12 languages)
- **Quality presets**: Fast (3B), Balanced (7B), Quality (14B), or API Server
- **Sound presets**: 6 synthesized tones (Soft Pop, Chime, Warm, Click, Marimba, Simple)
- **Pause/Resume**: Toggle dictation on and off without quitting
- **Launch at Login**: Auto-start when you turn on your Mac
- **Recent transcriptions**: Last 10 items, click to re-paste
- **100% private**: Everything runs locally. No data ever leaves your machine.

All settings persist between sessions.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- ~4GB RAM minimum (Fast preset), ~6GB recommended (Balanced)

## Installation

```bash
git clone https://github.com/0xbrando/dictate.git
cd dictate

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Models download automatically on first run (~2GB for Whisper, ~2-8GB for LLM depending on quality preset).

## Usage

```bash
source .venv/bin/activate
python -m dictate
```

| Action | Key |
|--------|-----|
| Record | Hold Left Control |
| Lock Recording | Press Space while holding Control |
| Stop Locked Recording | Press Control again |

macOS will prompt for Accessibility and Microphone permissions on first run.

### Menu Bar Options

All settings are accessible from the menu bar icon:

- **Pause/Resume Dictation** — stop listening without quitting
- **Microphone** — select input device
- **Quality** — choose model size (speed vs accuracy tradeoff)
- **Sounds** — pick recording start/stop tones
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
| API Server | ~250ms | 0 | Power users with a local LLM stack |
| Fast — 3B | ~250ms | 2GB | Quick cleanup, everyday use |
| Balanced — 7B | ~350ms | 5GB | Longer dictation, formal rewriting |
| Quality — 14B | ~500ms | 9GB | Best accuracy for bullet points and rewrites |

All times measured on Mac Studio M3 Ultra. Whisper transcription adds ~300ms.

### API Server Mode

If you run a local LLM server (vllm-mlx, LM Studio, Ollama, etc.), Dictate can use it instead of loading a bundled model — zero additional RAM. Point it at any OpenAI-compatible endpoint:

```bash
DICTATE_LLM_BACKEND=api DICTATE_LLM_API_URL=http://localhost:8005/v1/chat/completions python -m dictate
```

## How It Works

```
Mic → VAD → Whisper (local STT) → LLM (cleanup/rewrite) → Auto-paste
```

1. **Push-to-talk** captures audio via the microphone
2. **VAD** (voice activity detection) segments speech from silence
3. **Whisper Large V3 Turbo** transcribes locally via MLX
4. **Qwen** cleans up, rewrites, or converts to bullet points
5. **Auto-paste** puts the result into the focused window

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DICTATE_AUDIO_DEVICE` | Microphone device index | System default |
| `DICTATE_OUTPUT_MODE` | `type` or `clipboard` | `type` |
| `DICTATE_INPUT_LANGUAGE` | `auto`, `en`, `ja`, `ko`, etc. | `auto` |
| `DICTATE_OUTPUT_LANGUAGE` | Translation target (`auto` = same) | `auto` |
| `DICTATE_LLM_CLEANUP` | Enable LLM text cleanup | `true` |
| `DICTATE_LLM_MODEL` | `qwen`, `qwen-7b`, `qwen-14b` | `qwen` |
| `DICTATE_LLM_BACKEND` | `local` or `api` | `local` |
| `DICTATE_LLM_API_URL` | OpenAI-compatible endpoint | `http://localhost:8005/v1/chat/completions` |

## License

MIT — See [LICENSES.md](LICENSES.md) for dependency licenses.
