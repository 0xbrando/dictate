# macOS setup and inference review — September 2026

Reviewed against main `c4b1ed8`. This report covers the source fixes and local
validation, not a signed or notarized release build.

## Findings and repairs

- Updated FluidAudio 0.14.4 → 0.15.6 and built the Swift helper on this Mac.
- Missing Qwen runtime now falls back to Whisper with a compatible model and preserves the requested language. ANE fallback uses the Parakeet model explicitly.
- Raw/disabled cleanup no longer downloads or loads cleanup models.
- ANE startup allows first-run download/compilation time. Helper stderr no longer fills an unread pipe. Timeout or invalid JSON discards the helper, preventing stale output from being pasted on the next recording.
- Repetition filtering no longer drops the last word of odd-length partial repetitions.
- Local cleanup disables thinking through the chat template, and explicitly disables remote tokenizer code. Template/startup exceptions no longer strand the generation lock.
- Text API calls and model discovery enforce remote opt-in at the HTTP boundary, disable proxy inheritance, and reject redirects. These prevent a localhost configuration from redirecting a transcript elsewhere.
- Logs are owner-only and in an owner-only directory.
- App launches in foreground inside its bundle; corrected py2app package declarations (nested MLX modules were incorrectly treated as top-level packages; SciPy was excluded despite being required).
- Corrected Qwen language coverage and removed unsupported universal speed/unified-memory claims from the README.

- Fresh Hugging Face downloads now use a complete tqdm-compatible adapter, fixing a reproduced Xet `total` attribute failure. Verified a real Qwen3.5 2B download after the fix. Menu availability checks no longer import the native ML stack on the UI thread.

## Real local validation

Apple M2 Max, 32 GB, macOS 26.5.2. A synthetic English sample generated locally
with macOS `say` lasts 6.288 seconds. Both engines correctly returned:

> The quick brown fox jumps over the lazy dog. Please check that dictation works correctly on this Mac.

| Path | Measured time |
| --- | --- |
| FluidAudio 0.15.6 / Parakeet v3 | 128 ms, then 114 ms |
| Qwen3-ASR 0.6B 8-bit / mlx-audio 0.5.1 | 2.727 s first transcription, 364 ms warm |
| Qwen2.5 1.5B cleanup of a separate 24-word sample | 1.97 s |
| Qwen3.5 2B cleanup of the same sample | 2.22 s |

The ANE first download/load took 210.7 seconds. These are smoke tests, not a
representative accuracy benchmark or end-to-end microphone/hotkey/paste timing.
Both tested cleanup models retained filler/lower-case wording in that sample:
do not assume a successful inference means perfect editing.

Full test suite: **1,077 passed**. Dependency advisory scan: **no known
vulnerabilities among 93 checked packages**. This is not a guarantee of absence
of vulnerabilities. The repo-wide Ruff check reports 448 findings versus 454 on upstream main;
this is pre-existing style debt and it is not a clean lint gate. New boundary/regression files were checked.

## Models and current approach

The local STT → optional local cleanup design remains practical. Keep ANE
Parakeet v3 for English/its supported languages; Qwen3-ASR is a useful broader
multilingual alternative, already implemented and now locally verified. Whisper
remains a compatibility option. No evidence here supports a wholesale rewrite.

Sources checked directly:

- [FluidAudio 0.15.6 release](https://github.com/FluidInference/FluidAudio/releases/tag/v0.15.6): ASR/download fixes and newer optional streaming paths. Streaming is an optional future improvement for long recordings, not required for push-to-talk.
- [Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf): 30 languages and 22 Chinese dialects, not 52 distinct languages.
- [Apple SpeechAnalyzer](https://developer.apple.com/videos/play/wwdc2025/277/): native macOS 26 option worth a future comparative benchmark; not evaluated locally here.
- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B): newer cleanup candidate already available in the app; no comparative cleanup-quality win established by this audit.

## Permissions and release validation

Public model downloads need no Hugging Face login. The reviewed application
source makes no Keychain API calls. Default speech and cleanup inference are
local; model downloads and update checks use the network. Remote text API use
requires explicit opt-in.

Audio uses private temporary WAV files removed after processing. Recent
transcripts remain in memory. Output uses the system clipboard; clipboard
managers and Universal Clipboard may therefore see it. Accessibility is a
powerful OS permission used for global keyboard and paste behavior.

The local source-linked app reached `Pipeline ready`. A physical microphone,
push-to-talk, and paste test with the app's macOS permissions remains required.
The packaged app also needs clean-machine validation before a release is tagged.
