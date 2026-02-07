# Dictate

Push-to-talk voice dictation that lives in your macOS menu bar. 100% local — all processing runs on-device using Apple Silicon MLX models. No cloud, no API keys, no subscriptions.

## Features

- **Menu bar app** with reactive waveform icon that follows your voice
- **Push-to-talk**: Hold Control to record, release to transcribe
- **Lock recording**: Press Space while holding Control for hands-free
- **Auto-type**: Pastes directly into the focused window
- **LLM cleanup**: Fixes grammar and punctuation (Qwen 3B/7B/14B)
- **Translation**: Transcribe in one language, output in another (12 languages)
- **Quality presets**: Speed (3B), Balanced (7B), Quality (14B)
- **Microphone selection**: Pick any connected input device
- **Sound presets**: Customizable recording start/stop tones
- **Recent transcriptions**: Last 10 items, click to re-copy

All settings persist between sessions.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- ~4GB RAM minimum (Speed preset), ~6GB recommended (Balanced)

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

Or use the launcher script (add to Login Items for auto-start):
```bash
./start-menubar.sh
```

| Action | Key |
|--------|-----|
| Record | Hold Left Control |
| Lock Recording | Press Space while holding Control |
| Stop Locked Recording | Press Control again |
| Quit | Cmd+Q from menu |

macOS will prompt for Accessibility and Microphone permissions on first run.

## Configuration

Settings are available from the menu bar icon. For automation, use environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DICTATE_AUDIO_DEVICE` | Microphone device index | System default |
| `DICTATE_OUTPUT_MODE` | `type` or `clipboard` | `type` |
| `DICTATE_INPUT_LANGUAGE` | `auto`, `en`, `ja`, `ko`, etc. | `auto` |
| `DICTATE_OUTPUT_LANGUAGE` | Translation target (`auto` = same) | `auto` |
| `DICTATE_LLM_CLEANUP` | Enable LLM text cleanup | `true` |
| `DICTATE_LLM_MODEL` | `qwen`, `qwen-7b`, `qwen-14b` | `qwen` |

## How It Works

```
Microphone -> VAD -> Whisper (transcription) -> Qwen (cleanup) -> Paste into window
```

1. **Push-to-talk** captures audio via the microphone
2. **VAD** (voice activity detection) segments speech from silence
3. **Whisper Large V3 Turbo** transcribes locally via MLX
4. **Qwen** fixes grammar, punctuation, and optionally translates
5. **Auto-type** pastes the result into the focused window

## RAM Usage

| Preset | LLM Model | Total RAM |
|--------|-----------|-----------|
| Speed | Qwen 3B | ~4GB |
| Balanced | Qwen 7B | ~6GB |
| Quality | Qwen 14B | ~10GB |

## License

MIT — See [LICENSES.md](LICENSES.md) for dependency licenses.
