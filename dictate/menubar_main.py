"""Entry point for the menu bar app: python -m dictate.menubar_main"""

import platform
import subprocess
import sys

if platform.system() != "Darwin":
    print("Dictate requires macOS with Apple Silicon. See https://github.com/0xbrando/dictate", file=sys.stderr)
    sys.exit(1)

import fcntl
import logging
import os
import signal
import time
from pathlib import Path

# Disable HuggingFace telemetry — all inference is local
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

# Load .env file if it exists (before importing config)
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

LOCK_FILE = Path.home() / "Library" / "Application Support" / "Dictate" / "dictate.lock"
LOG_FILE = Path.home() / "Library" / "Logs" / "Dictate" / "dictate.log"


def _config_command(args: list[str]) -> int:
    """Handle 'dictate config' subcommands.

    Usage:
        dictate config              Show all current settings
        dictate config show         Show all current settings
        dictate config set KEY VAL  Set a preference
        dictate config reset        Reset all preferences to defaults
        dictate config path         Show config file path
    """
    from dictate.presets import (
        Preferences, QUALITY_PRESETS, STT_PRESETS, WRITING_STYLES,
        INPUT_LANGUAGES, OUTPUT_LANGUAGES, PTT_KEYS, COMMAND_KEYS,
        SOUND_PRESETS, PREFS_DIR, PREFS_FILE,
    )

    W = "\033[97m"   # bright white
    G = "\033[32m"   # green
    Y = "\033[33m"   # yellow
    R = "\033[31m"   # red
    D = "\033[2m"    # dim
    B = "\033[1m"    # bold
    N = "\033[0m"    # reset

    # Valid config keys and their accepted values
    CONFIG_KEYS: dict[str, dict] = {
        "writing_style": {
            "description": "Writing style for LLM cleanup",
            "values": [s[0] for s in WRITING_STYLES],
            "type": "choice",
        },
        "quality": {
            "description": "Quality preset (LLM model size)",
            "values": list(range(len(QUALITY_PRESETS))),
            "aliases": {
                "api": 0, "speedy": 1, "fast": 2, "balanced": 3, "quality": 4,
            },
            "type": "int_or_alias",
        },
        "stt": {
            "description": "Speech-to-text engine",
            "values": list(range(len(STT_PRESETS))),
            "aliases": {p.engine.value: i for i, p in enumerate(STT_PRESETS)},
            "type": "int_or_alias",
        },
        "input_language": {
            "description": "Input language for transcription",
            "values": [lang[0] for lang in INPUT_LANGUAGES],
            "type": "choice",
        },
        "output_language": {
            "description": "Output language (translation target)",
            "values": [lang[0] for lang in OUTPUT_LANGUAGES],
            "type": "choice",
        },
        "ptt_key": {
            "description": "Push-to-talk key",
            "values": [k[0] for k in PTT_KEYS],
            "type": "choice",
        },
        "command_key": {
            "description": "Command key (lock recording)",
            "values": [k[0] for k in COMMAND_KEYS],
            "type": "choice",
        },
        "llm_cleanup": {
            "description": "Enable LLM text cleanup",
            "values": ["on", "off", "true", "false", "1", "0"],
            "type": "bool",
        },
        "sound": {
            "description": "Sound feedback preset",
            "values": list(range(len(SOUND_PRESETS))),
            "aliases": {p.style: i for i, p in enumerate(SOUND_PRESETS)},
            "type": "int_or_alias",
        },
        "llm_endpoint": {
            "description": "LLM API endpoint (host:port)",
            "type": "string",
        },
        "advanced_mode": {
            "description": "Enable advanced mode in menu bar",
            "values": ["on", "off", "true", "false", "1", "0"],
            "type": "bool",
        },
    }

    subcmd = args[0] if args else "show"

    if subcmd == "path":
        print(str(PREFS_FILE))
        return 0

    if subcmd == "reset":
        prefs = Preferences()
        prefs.save()
        print(f"{G}✓{N} Preferences reset to defaults")
        return 0

    if subcmd == "set":
        if len(args) < 3:
            print(f"{R}Usage:{N} dictate config set KEY VALUE", file=sys.stderr)
            print(f"\n{W}Available keys:{N}")
            for key, info in CONFIG_KEYS.items():
                desc = info["description"]
                if info["type"] == "choice":
                    vals = ", ".join(str(v) for v in info["values"])
                    print(f"  {Y}{key}{N}  {D}{desc}{N}  [{vals}]")
                elif info["type"] == "int_or_alias":
                    aliases = info.get("aliases", {})
                    vals = ", ".join(str(v) for v in info["values"])
                    alias_str = ", ".join(aliases.keys()) if aliases else ""
                    print(f"  {Y}{key}{N}  {D}{desc}{N}  [{vals}] or [{alias_str}]")
                elif info["type"] == "bool":
                    print(f"  {Y}{key}{N}  {D}{desc}{N}  [on/off]")
                else:
                    print(f"  {Y}{key}{N}  {D}{desc}{N}")
            return 1

        key = args[1]
        value = args[2]

        if key not in CONFIG_KEYS:
            print(f"{R}Unknown key:{N} {key}", file=sys.stderr)
            print(f"{D}Available keys: {', '.join(CONFIG_KEYS.keys())}{N}", file=sys.stderr)
            return 1

        prefs = Preferences.load()
        info = CONFIG_KEYS[key]

        # Resolve the value based on type
        if info["type"] == "choice":
            if value not in info["values"]:
                print(f"{R}Invalid value:{N} {value}", file=sys.stderr)
                print(f"{D}Valid values: {', '.join(str(v) for v in info['values'])}{N}", file=sys.stderr)
                return 1
            resolved = value

        elif info["type"] == "int_or_alias":
            aliases = info.get("aliases", {})
            if value in aliases:
                resolved = aliases[value]
            elif value.isdigit() and int(value) in info["values"]:
                resolved = int(value)
            else:
                valid = list(str(v) for v in info["values"]) + list(aliases.keys())
                print(f"{R}Invalid value:{N} {value}", file=sys.stderr)
                print(f"{D}Valid values: {', '.join(valid)}{N}", file=sys.stderr)
                return 1

        elif info["type"] == "bool":
            if value.lower() in ("on", "true", "1"):
                resolved = True
            elif value.lower() in ("off", "false", "0"):
                resolved = False
            else:
                print(f"{R}Invalid value:{N} {value}. Use on/off", file=sys.stderr)
                return 1

        elif info["type"] == "string":
            resolved = value

        else:
            resolved = value

        # Map config key to Preferences field name
        field_map = {
            "quality": "quality_preset",
            "stt": "stt_preset",
            "sound": "sound_preset",
        }
        field_name = field_map.get(key, key)
        setattr(prefs, field_name, resolved)
        prefs.save()
        print(f"{G}✓{N} Set {W}{key}{N} = {Y}{value}{N}")
        return 0

    # Default: show config
    if subcmd not in ("show", "list", "get"):
        print(f"{R}Unknown subcommand:{N} {subcmd}", file=sys.stderr)
        print(f"{D}Usage: dictate config [show|set|reset|path]{N}", file=sys.stderr)
        return 1

    if not PREFS_FILE.exists():
        print(f"{D}No preferences file. Using defaults.{N}")
        prefs = Preferences()
    else:
        prefs = Preferences.load()

    # Display current config
    quality = QUALITY_PRESETS[prefs.quality_preset] if prefs.quality_preset < len(QUALITY_PRESETS) else None
    stt = STT_PRESETS[prefs.stt_preset] if prefs.stt_preset < len(STT_PRESETS) else None
    style_name = next((s[1] for s in WRITING_STYLES if s[0] == prefs.writing_style), prefs.writing_style)
    sound = SOUND_PRESETS[prefs.sound_preset] if prefs.sound_preset < len(SOUND_PRESETS) else None
    ptt_label = next((k[1] for k in PTT_KEYS if k[0] == prefs.ptt_key), prefs.ptt_key)
    cmd_label = next((k[1] for k in COMMAND_KEYS if k[0] == prefs.command_key), prefs.command_key)
    in_lang = next((l[1] for l in INPUT_LANGUAGES if l[0] == prefs.input_language), prefs.input_language)
    out_lang = next((l[1] for l in OUTPUT_LANGUAGES if l[0] == prefs.output_language), prefs.output_language)

    print(f"\n{W}{B}Dictate Config{N}\n")
    print(f"  {W}writing_style{N}   {Y}{prefs.writing_style}{N}  {D}({style_name}){N}")
    print(f"  {W}quality{N}         {Y}{prefs.quality_preset}{N}  {D}({quality.label if quality else 'Unknown'}){N}")
    print(f"  {W}stt{N}             {Y}{prefs.stt_preset}{N}  {D}({stt.label if stt else 'Unknown'}){N}")
    print(f"  {W}input_language{N}  {Y}{prefs.input_language}{N}  {D}({in_lang}){N}")
    print(f"  {W}output_language{N} {Y}{prefs.output_language}{N}  {D}({out_lang}){N}")
    print(f"  {W}ptt_key{N}         {Y}{prefs.ptt_key}{N}  {D}({ptt_label}){N}")
    print(f"  {W}command_key{N}     {Y}{prefs.command_key}{N}  {D}({cmd_label}){N}")
    print(f"  {W}llm_cleanup{N}     {Y}{'on' if prefs.llm_cleanup else 'off'}{N}")
    print(f"  {W}sound{N}           {Y}{prefs.sound_preset}{N}  {D}({sound.label if sound else 'Unknown'}){N}")
    print(f"  {W}llm_endpoint{N}    {Y}{prefs.llm_endpoint}{N}")
    print(f"  {W}advanced_mode{N}   {Y}{'on' if prefs.advanced_mode else 'off'}{N}")
    print()
    print(f"  {D}Config file: {PREFS_FILE}{N}")
    print(f"  {D}Use 'dictate config set KEY VALUE' to change a setting{N}")
    print()
    return 0


def _show_status() -> int:
    """Show system info and model status for troubleshooting."""
    from dictate import __version__
    from dictate.config import is_model_cached, get_cached_model_disk_size, WHISPER_MODEL
    from dictate.presets import (
        Preferences, QUALITY_PRESETS, STT_PRESETS, WRITING_STYLES,
        PREFS_DIR, PREFS_FILE,
    )

    W = "\033[97m"   # bright white
    G = "\033[32m"   # green
    R = "\033[31m"   # red
    D = "\033[2m"    # dim
    B = "\033[1m"    # bold
    N = "\033[0m"    # reset

    print(f"\n{W}{B}Dictate Status{N}  {D}v{__version__}{N}\n")

    # System info
    import platform
    chip = platform.processor() or "unknown"
    mac_ver = platform.mac_ver()[0] or "unknown"
    py_ver = platform.python_version()
    print(f"  {W}System{N}")
    print(f"  macOS {mac_ver} · Python {py_ver} · {chip}")
    print()

    # Check if running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "dictate.menubar_main"],
            capture_output=True, text=True, check=False,
        )
        pids = [p for p in result.stdout.strip().split("\n") if p and p != str(os.getpid())]
        if pids:
            print(f"  {W}Status{N}  {G}● Running{N} (PID {pids[0]})")
        else:
            print(f"  {W}Status{N}  {D}○ Not running{N}")
    except Exception:
        print(f"  {W}Status{N}  {D}? Unknown{N}")
    print()

    # Models
    print(f"  {W}Models{N}")
    whisper_cached = is_model_cached(WHISPER_MODEL)
    whisper_size = get_cached_model_disk_size(WHISPER_MODEL) if whisper_cached else "not downloaded"
    st = f"{G}✓{N}" if whisper_cached else f"{R}✗{N}"
    print(f"  {st} Whisper: {WHISPER_MODEL} ({whisper_size})")

    # Check Parakeet
    try:
        import parakeet_mlx  # noqa: F401
        print(f"  {G}✓{N} Parakeet: installed")
    except ImportError:
        print(f"  {D}○ Parakeet: not installed{N}")

    # LLM models
    from dictate.config import LLMModel
    for model in LLMModel:
        cached = is_model_cached(model.hf_repo)
        size = get_cached_model_disk_size(model.hf_repo) if cached else "not downloaded"
        st = f"{G}✓{N}" if cached else f"{D}○{N}"
        print(f"  {st} LLM {model.value}: {model.hf_repo} ({size})")
    print()

    # Preferences
    print(f"  {W}Preferences{N}")
    print(f"  Config dir: {PREFS_DIR}")
    if PREFS_FILE.exists():
        prefs = Preferences.load()
        quality = QUALITY_PRESETS[prefs.quality_preset] if prefs.quality_preset < len(QUALITY_PRESETS) else None
        stt = STT_PRESETS[prefs.stt_preset] if prefs.stt_preset < len(STT_PRESETS) else None
        style_name = next((s[1] for s in WRITING_STYLES if s[0] == prefs.writing_style), prefs.writing_style)
        print(f"  Quality: {quality.label if quality else 'Unknown'}")
        print(f"  STT: {stt.label if stt else 'Unknown'}")
        print(f"  Writing style: {style_name}")
        print(f"  LLM cleanup: {'on' if prefs.llm_cleanup else 'off'}")
        print(f"  Input language: {prefs.input_language}")
        print(f"  Output language: {prefs.output_language}")
        print(f"  PTT key: {prefs.ptt_key}")
    else:
        print(f"  {D}No preferences file (will use defaults){N}")
    print()

    # Log file
    if LOG_FILE.exists():
        size = LOG_FILE.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  {W}Logs{N}")
        print(f"  {LOG_FILE} ({size_str})")
    print()

    return 0


def _show_stats() -> int:
    """Show usage statistics."""
    from dictate.stats import UsageStats

    W = "\033[97m"   # bright white
    Y = "\033[33m"   # yellow
    D = "\033[2m"    # dim
    B = "\033[1m"    # bold
    N = "\033[0m"    # reset

    stats = UsageStats.load()

    if stats.total_dictations == 0:
        print(f"\n{D}No dictations recorded yet. Start talking!{N}\n")
        return 0

    print(f"\n{W}{B}Dictate Stats{N}\n")
    print(f"  {W}Dictations{N}      {Y}{stats.total_dictations:,}{N}")
    print(f"  {W}Words{N}           {Y}{stats.total_words:,}{N}")
    print(f"  {W}Characters{N}      {Y}{stats.total_characters:,}{N}")
    print(f"  {W}Audio recorded{N}  {Y}{stats.format_duration(stats.total_audio_seconds)}{N}")
    print()

    # Average words per dictation
    avg_words = stats.total_words / stats.total_dictations
    print(f"  {W}Avg words/dictation{N}  {Y}{avg_words:.1f}{N}")

    # Time info
    print(f"  {W}First use{N}       {D}{stats.format_time_ago(stats.first_use)}{N}")
    print(f"  {W}Last use{N}        {D}{stats.format_time_ago(stats.last_use)}{N}")

    # Writing styles breakdown
    if stats.styles_used:
        print(f"\n  {W}Writing Styles{N}")
        for style, count in sorted(stats.styles_used.items(), key=lambda x: -x[1]):
            pct = (count / stats.total_dictations) * 100
            bar = "█" * max(1, int(pct / 5))
            print(f"  {D}{bar}{N} {Y}{style}{N} {D}({count}, {pct:.0f}%){N}")
    print()
    return 0


def _run_doctor() -> int:
    """Run diagnostic checks and report issues."""
    from dictate import __version__
    from dictate.config import is_model_cached, WHISPER_MODEL

    W = "\033[97m"
    G = "\033[32m"
    Y = "\033[33m"
    R = "\033[31m"
    D = "\033[2m"
    B = "\033[1m"
    N = "\033[0m"

    print(f"\n{W}{B}Dictate Doctor{N}  {D}v{__version__}{N}\n")

    issues = []
    warnings = []

    # 1. Check macOS version
    import platform
    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        major = int(mac_ver.split(".")[0])
        if major >= 14:
            print(f"  {G}✓{N} macOS {mac_ver} (supported)")
        elif major >= 13:
            print(f"  {Y}⚠{N} macOS {mac_ver} (may work, 14+ recommended)")
            warnings.append("macOS 14+ recommended for best MLX performance")
        else:
            print(f"  {R}✗{N} macOS {mac_ver} (too old — need 13+)")
            issues.append("macOS 13+ required for Apple Silicon MLX")
    else:
        print(f"  {Y}?{N} Could not detect macOS version")

    # 2. Check Apple Silicon
    chip = platform.processor()
    if "arm" in chip.lower():
        print(f"  {G}✓{N} Apple Silicon ({chip})")
    else:
        print(f"  {R}✗{N} Processor: {chip} (Apple Silicon required)")
        issues.append("Apple Silicon required for MLX inference")

    # 3. Check Python version
    py_ver = platform.python_version()
    py_major, py_minor = int(py_ver.split(".")[0]), int(py_ver.split(".")[1])
    if py_major == 3 and 11 <= py_minor <= 13:
        print(f"  {G}✓{N} Python {py_ver}")
    elif py_major == 3 and py_minor >= 14:
        print(f"  {Y}⚠{N} Python {py_ver} (some dependencies may not support 3.14+)")
        warnings.append("Python 3.14+ has limited package support")
    else:
        print(f"  {R}✗{N} Python {py_ver} (need 3.11+)")
        issues.append("Python 3.11+ required")

    # 4. Check microphone access
    print()
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if input_devices:
            default_input = sd.query_devices(kind="input")
            name = default_input.get("name", "Unknown")
            print(f"  {G}✓{N} Microphone: {name}")
            print(f"    {D}{len(input_devices)} input device(s) available{N}")
        else:
            print(f"  {R}✗{N} No input devices found")
            issues.append("No microphone detected — check System Settings > Sound")
    except Exception as e:
        print(f"  {Y}⚠{N} Could not check microphone: {e}")
        warnings.append("Could not query audio devices — sounddevice may not be installed")

    # 5. Check Whisper model
    print()
    if is_model_cached(WHISPER_MODEL):
        print(f"  {G}✓{N} Whisper model cached")
    else:
        print(f"  {Y}⚠{N} Whisper model not downloaded yet")
        warnings.append(f"Whisper model ({WHISPER_MODEL}) will download on first use (~1.5 GB)")

    # 6. Check Parakeet
    try:
        import parakeet_mlx  # noqa: F401
        print(f"  {G}✓{N} Parakeet STT available")
    except ImportError:
        print(f"  {D}○{N} Parakeet not installed (optional — Whisper is default)")

    # 7. Check LLM endpoint (if configured for API mode)
    from dictate.presets import Preferences, PREFS_FILE
    if PREFS_FILE.exists():
        prefs = Preferences.load()
        if prefs.llm_endpoint and prefs.llm_endpoint != "localhost:8080":
            import urllib.request
            import urllib.error
            url = f"http://{prefs.llm_endpoint}/v1/models"
            try:
                req = urllib.request.urlopen(url, timeout=3)
                req.close()
                print(f"  {G}✓{N} LLM endpoint reachable: {prefs.llm_endpoint}")
            except (urllib.error.URLError, OSError):
                print(f"  {Y}⚠{N} LLM endpoint not reachable: {prefs.llm_endpoint}")
                warnings.append(f"LLM endpoint ({prefs.llm_endpoint}) is not responding")

    # 8. Check disk space
    print()
    import shutil
    total, used, free = shutil.disk_usage(Path.home())
    free_gb = free / (1024 ** 3)
    if free_gb > 10:
        print(f"  {G}✓{N} Disk space: {free_gb:.1f} GB free")
    elif free_gb > 3:
        print(f"  {Y}⚠{N} Disk space: {free_gb:.1f} GB free (getting low)")
        warnings.append("Less than 10 GB free — model downloads may fail")
    else:
        print(f"  {R}✗{N} Disk space: {free_gb:.1f} GB free (critically low)")
        issues.append("Less than 3 GB free — Dictate needs space for models")

    # 9. Check accessibility permissions
    print()
    try:
        # Check if we can detect key events (rough proxy for accessibility)
        import Quartz  # noqa: F401
        print(f"  {G}✓{N} Quartz framework available (key detection)")
    except ImportError:
        print(f"  {Y}⚠{N} Quartz not available — key detection may not work")
        warnings.append("Quartz framework not available")

    # 10. Check for duplicate instances
    try:
        result = subprocess.run(
            ["pgrep", "-f", "dictate.menubar_main"],
            capture_output=True, text=True, check=False,
        )
        pids = [p for p in result.stdout.strip().split("\n") if p and p != str(os.getpid())]
        if len(pids) > 1:
            print(f"  {Y}⚠{N} Multiple instances running (PIDs: {', '.join(pids)})")
            warnings.append("Multiple Dictate instances — run 'pkill -f dictate.menubar_main' to clean up")
        elif len(pids) == 1:
            print(f"  {G}✓{N} One instance running (PID {pids[0]})")
        else:
            print(f"  {D}○{N} Dictate not currently running")
    except Exception:
        pass

    # Summary
    print()
    if issues:
        print(f"  {R}{B}Issues ({len(issues)}):{N}")
        for issue in issues:
            print(f"  {R}  • {issue}{N}")
    if warnings:
        print(f"  {Y}{B}Warnings ({len(warnings)}):{N}")
        for warning in warnings:
            print(f"  {Y}  • {warning}{N}")
    if not issues and not warnings:
        print(f"  {G}{B}All checks passed!{N} Dictate should work correctly.")
    elif not issues:
        print(f"\n  {G}No critical issues.{N} Warnings are informational.")
    else:
        print(f"\n  {R}Fix the issues above before running Dictate.{N}")
    print()

    return 1 if issues else 0


def _list_devices() -> int:
    """List available audio input devices."""
    W = "\033[97m"
    G = "\033[32m"
    Y = "\033[33m"
    D = "\033[2m"
    B = "\033[1m"
    N = "\033[0m"

    print(f"\n{W}{B}Audio Input Devices{N}\n")

    try:
        from dictate.audio import list_input_devices
        devices = list_input_devices()
    except ImportError:
        print(f"  {Y}⚠{N} sounddevice not installed")
        return 1
    except Exception as e:
        print(f"  {Y}⚠{N} Could not query devices: {e}")
        return 1

    if not devices:
        print(f"  {D}No input devices found.{N}")
        print(f"  {D}Check System Settings > Sound > Input.{N}")
        return 1

    for dev in devices:
        if dev.is_default:
            print(f"  {G}●{N} [{W}{dev.index}{N}] {dev.name}  {G}← default{N}")
        else:
            print(f"  {D}○{N} [{W}{dev.index}{N}] {dev.name}")

    print()
    print(f"  {D}Set input device: dictate config set device_id <NUMBER>{N}")
    print(f"  {D}Reset to default: dictate config set device_id auto{N}")
    print()
    return 0


def _run_update(check_only: bool = False, from_github: bool = False) -> int:
    """Update Dictate to the latest version.

    Tries PyPI first (``pip install --upgrade dictate-mlx``).  If PyPI
    fails (package not published yet), falls back to installing from
    the GitHub repository.

    Args:
        check_only: Only check for a newer version — do not install.
        from_github: Force install from GitHub instead of PyPI.
    """
    from dictate import __version__

    W = "\033[97m"
    G = "\033[32m"
    Y = "\033[33m"
    R = "\033[31m"
    D = "\033[2m"
    B = "\033[1m"
    N = "\033[0m"

    GITHUB_REPO = "https://github.com/0xbrando/dictate.git"

    print(f"\n{W}{B}Dictate Update{N}  {D}current: v{__version__}{N}\n")

    if check_only:
        print(f"  {D}Checking for updates...{N}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "index", "versions", "dictate-mlx"],
                capture_output=True, text=True, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse latest version from pip index output
                for line in result.stdout.strip().split("\n"):
                    if "Available versions:" in line or "LATEST:" in line:
                        print(f"  {G}PyPI:{N} {line.strip()}")
                        break
                else:
                    print(f"  {D}PyPI: {result.stdout.strip()}{N}")
            else:
                print(f"  {Y}Not on PyPI yet.{N} Install from GitHub:")
                print(f"  {D}pip install git+{GITHUB_REPO}{N}")
        except Exception:
            print(f"  {Y}Could not check PyPI.{N}")
        print(f"\n  {D}Run 'dictate update' to install the latest version.{N}\n")
        return 0

    # --- Install ---
    if from_github:
        print(f"  {W}Installing from GitHub...{N}")
        install_cmd = [
            sys.executable, "-m", "pip", "install", "--upgrade",
            f"git+{GITHUB_REPO}",
        ]
    else:
        # Try PyPI first
        print(f"  {W}Checking PyPI...{N}")
        install_cmd = [
            sys.executable, "-m", "pip", "install", "--upgrade", "dictate-mlx",
        ]

    try:
        result = subprocess.run(
            install_cmd, capture_output=True, text=True, check=False,
        )

        if result.returncode != 0 and not from_github:
            # PyPI failed — try GitHub
            print(f"  {Y}Not on PyPI yet — installing from GitHub...{N}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade",
                 f"git+{GITHUB_REPO}"],
                capture_output=True, text=True, check=False,
            )

        if result.returncode != 0:
            print(f"\n  {R}✗ Update failed{N}")
            if result.stderr:
                # Show last few lines of error
                err_lines = result.stderr.strip().split("\n")[-3:]
                for line in err_lines:
                    print(f"  {D}{line}{N}")
            return 1

        # Check new version
        ver_result = subprocess.run(
            [sys.executable, "-c", "from dictate import __version__; print(__version__)"],
            capture_output=True, text=True, check=False,
        )
        new_version = ver_result.stdout.strip() if ver_result.returncode == 0 else "unknown"

        if new_version == __version__:
            print(f"\n  {G}✓ Already up to date{N} (v{__version__})")
        else:
            print(f"\n  {G}✓ Updated{N} v{__version__} → v{new_version}")

    except Exception as e:
        print(f"\n  {R}✗ Update failed:{N} {e}")
        return 1

    # Kill any running menu bar instance
    try:
        result = subprocess.run(
            ["pgrep", "-f", "dictate.menubar_main"],
            capture_output=True, text=True, check=False,
        )
        pids = [p for p in result.stdout.strip().split("\n") if p and p != str(os.getpid())]
        if pids:
            print(f"  {D}Stopping running instance (PID {pids[0]})...{N}")
            subprocess.run(
                ["pkill", "-f", "dictate\\.menubar_main"],
                capture_output=True, check=False,
            )
            time.sleep(0.5)

            # Relaunch
            try:
                subprocess.Popen(
                    [sys.executable, "-m", "dictate.menubar_main"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                print(f"  {G}✓ Restarted{N}")
            except Exception:
                print(f"  {Y}Run 'dictate' to restart.{N}")
    except Exception:
        pass

    print()
    
    return 0


def _acquire_singleton_lock() -> int | None:
    """Acquire an exclusive lock to prevent duplicate instances.

    Returns the lock fd on success, or None if another instance is running.
    The fd must be kept open for the lifetime of the process.
    """
    import errno
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.ftruncate(fd, os.lseek(fd, 0, os.SEEK_CUR))
        return fd
    except OSError as e:
        os.close(fd)
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
            return None  # Another instance holds the lock
        # Unexpected OS error — log it so the user can debug
        print(f"Lock file error: {e}", file=sys.stderr)
        return None


def _daemonize() -> None:
    """Re-launch as a detached background process.

    os.fork() is incompatible with macOS AppKit/ObjC — the forked child
    crashes with objc_initializeAfterForkError.  Instead we spawn a fresh
    subprocess with --foreground so the child runs the app directly.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(LOG_FILE, "a")
    try:
        subprocess.Popen(
            [sys.executable, "-m", "dictate.menubar_main", "--foreground"],
            stdout=log_fd,
            stderr=log_fd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        print(f"Background launch failed ({e}), running in foreground", file=sys.stderr)
        log_fd.close()
        return
    log_fd.close()
    os._exit(0)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("urllib3", "httpx", "mlx", "transformers", "tokenizers", "sounddevice"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> int:
    # Handle --version flag
    if "--version" in sys.argv or "-V" in sys.argv:
        from dictate import __version__
        print(f"dictate {__version__}")
        return 0

    # Handle --help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        from dictate import __version__
        print(f"dictate v{__version__} — push-to-talk voice dictation for macOS")
        print()
        print("Usage: dictate [COMMAND] [OPTIONS]")
        print()
        print("Commands:")
        print("  (default)       Launch Dictate in the menu bar")
        print("  config          View and modify preferences")
        print("  stats           Show usage statistics")
        print("  status          Show system info and model status")
        print("  doctor          Run diagnostic checks")
        print("  devices         List audio input devices")
        print("  update          Update to the latest version")
        print("  update --check  Check for updates without installing")
        print("  update --github Install latest from GitHub")
        print()
        print("Options:")
        print("  -f, --foreground  Run in foreground (show logs)")
        print("  -V, --version     Show version and exit")
        print("  -h, --help        Show this help and exit")
        print()
        print("Config examples:")
        print("  dictate config                    Show current settings")
        print("  dictate config set writing_style email")
        print("  dictate config set quality speedy")
        print("  dictate config set ptt_key cmd_r")
        print("  dictate config reset              Reset to defaults")
        print()
        print("https://github.com/0xbrando/dictate")
        return 0

    # Handle config command
    if "config" in sys.argv:
        config_idx = sys.argv.index("config")
        return _config_command(sys.argv[config_idx + 1:])

    # Handle stats command
    if "stats" in sys.argv:
        return _show_stats()

    # Handle status command
    if "status" in sys.argv:
        return _show_status()

    # Handle doctor command
    if "doctor" in sys.argv:
        return _run_doctor()

    # Handle devices command
    if "devices" in sys.argv:
        return _list_devices()

    # Handle update command
    if "update" in sys.argv or "--update" in sys.argv:
        check_only = "--check" in sys.argv
        from_github = "--github" in sys.argv
        return _run_update(check_only=check_only, from_github=from_github)
    
    # Show banner by writing directly to /dev/tty so it always appears in the
    # terminal even when stdout/stderr are redirected (e.g. via nohup).
    foreground = "--foreground" in sys.argv or "-f" in sys.argv
    try:
        _tty = open("/dev/tty", "w")
    except OSError:
        _tty = None

    if _tty is not None:
        try:
            from dictate import __version__
            Y = "\033[33m"    # yellow/dark orange
            O = "\033[93m"    # bright yellow/orange
            W = "\033[97m"    # bright white
            D = "\033[2m"     # dim
            B = "\033[1m"     # bold
            R = "\033[0m"     # reset
            _tty.write(f"""
{O}       ___      __        __
{O}  ____/ (_)____/ /_____ _/ /____
{Y} / __  / / ___/ __/ __ `/ __/ _ \\
{Y}/ /_/ / / /__/ /_/ /_/ / /_/  __/
{Y}\\__,_/_/\\___/\\__/\\__,_/\\__/\\___/{R}

  {W}{B}speak. it types.{R}  {D}v{__version__} · 100% local{R}

  {D}Dictate is now running in your menu bar.
  You can close this terminal — it won't stop the app.{R}

  {W}HOW TO USE{R}
  {D}Hold{R} {O}Left Ctrl{R}       {D}talk, release to transcribe{R}
  {D}Hold{R} {O}Ctrl + Space{R}    {D}lock recording (hands-free){R}
  {D}Tap{R}  {O}Ctrl{R}            {D}to stop locked recording{R}
  {D}Change the key, model, and more from the menu bar icon.{R}

  {W}TIPS{R}
  {D}Parakeet is English-only. Switch to Whisper for other languages
  under Advanced → STT Engine.{R}
  {D}Writing styles (Clean, Formal, Bullets) change how your text
  is polished — find them in the menu bar.{R}
  {D}Add names, slang, or technical terms to your personal dictionary
  so they're always spelled right — Advanced → Dictionary.{R}

  {W}COMMANDS{R}
  {O}dictate{R}          {D}launch dictate{R}
  {O}dictate update{R}   {D}update to the latest version{R}
  {O}dictate -f{R}       {D}run in foreground (debug){R}
""")
            _tty.flush()
        except Exception:
            pass
        finally:
            _tty.close()

    # Daemonize if not already in foreground mode
    if not foreground:
        _daemonize()

    setup_logging()
    logger = logging.getLogger(__name__)

    lock_fd = _acquire_singleton_lock()
    if lock_fd is None:
        logger.error("Another instance of Dictate is already running. Exiting.")
        print("Dictate is already running.", file=sys.stderr)
        return 1

    logger.info("Starting Dictate menu bar app (pid=%d)", os.getpid())

    try:
        from dictate.menubar import DictateMenuBarApp

        app = DictateMenuBarApp()
        app.start_app()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception:
        logger.exception("Fatal error")
        return 1
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    sys.exit(main())
