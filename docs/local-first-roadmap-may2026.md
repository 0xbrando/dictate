# Local-First Roadmap: May 2026

Dictate's public positioning should stay simple: local dictation for macOS, no cloud account, no API key, no subscription. The implementation should follow the same hierarchy.

## Recommended STT Stack

1. **FluidAudio / Parakeet TDT v3 on ANE**
   - Primary default for supported languages.
   - Runs through Core ML on Apple Neural Engine, keeping GPU memory free for text cleanup.
   - Best fit for a native Swift/SwiftUI app shell.

2. **Qwen3-ASR on MLX**
   - Best local multilingual fallback.
   - Covers 52 languages/dialects and is the right path for CJK, Arabic, Hindi, and other languages outside Parakeet's European-language set.
   - Keep optional because it adds another runtime dependency.

3. **Parakeet TDT v3 on MLX**
   - Practical fallback when the Swift/ANE helper is missing.
   - Useful for European languages and systems where the user has not built `swift-stt`.

4. **Whisper Large V3 Turbo**
   - Compatibility fallback for maximum language coverage.
   - Do not position as the lead engine anymore.

## TTS Direction

TTS is not core to dictation, but it is relevant if Dictate grows into an agent voice layer. If added, keep it local-first:

- **FluidAudio Kokoro / PocketTTS** for the native Swift path.
- **Kokoro 82M** for lightweight local speech output.
- **Chatterbox, Dia, Orpheus** only for optional expressive voice experiments, not the default app path.

## UI Direction

The current Python menu bar app is good enough for the ML pipeline, but public onboarding will be cleaner with a native shell:

- SwiftUI menu bar app for permissions, settings, status, model downloads, and diagnostics.
- Keep Python/MLX as a subprocess bridge initially.
- Keep `swift-stt` as the first fully native engine.
- Move pieces to Swift incrementally instead of doing a risky full rewrite.

## Cloud Policy

Cloud should stay opt-in. The default install should not ask for API keys or send audio/text off-machine.

- Local models are the default.
- Localhost OpenAI-compatible LLM servers are allowed.
- Remote endpoints require `DICTATE_ALLOW_REMOTE_API=1`.
- Docs should call this out directly so users understand the privacy boundary.
