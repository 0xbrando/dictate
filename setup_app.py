"""py2app build configuration for Dictate.app

Build a standalone macOS .app bundle:
    python setup_app.py py2app          # production build
    python setup_app.py py2app -A       # development build (symlinked)

Output: dist/Dictate.app
"""

from setuptools import setup

APP = ["dictate/menubar_main.py"]
DATA_FILES = []
OPTIONS = {
    "argv_emulation": False,  # Must be False for menu bar apps
    "iconfile": "assets/dictate.icns",
    "plist": {
        "CFBundleName": "Dictate",
        "CFBundleDisplayName": "Dictate",
        "CFBundleIdentifier": "com.0xbrando.dictate",
        "CFBundleVersion": "2.4.1",
        "CFBundleShortVersionString": "2.4.1",
        "LSUIElement": True,  # Menu bar only — no Dock icon
        "LSMinimumSystemVersion": "14.0",  # macOS Sonoma+ for MLX
        "NSMicrophoneUsageDescription": (
            "Dictate needs microphone access for voice-to-text transcription. "
            "All processing happens locally on your device."
        ),
        "NSAppleEventsUsageDescription": (
            "Dictate uses accessibility to type transcribed text "
            "into your active application."
        ),
    },
    "packages": [
        "dictate",
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx_whisper",
        "mlx_lm",
        "numpy",
        "sounddevice",
        "pynput",
        "pyperclip",
        "huggingface_hub",
        "safetensors",
        "tokenizers",
        "transformers",
        "rumps",
        "dotenv",
    ],
    "includes": [
        "objc",
        "AppKit",
        "Foundation",
    ],
    "excludes": [
        "tkinter",
        "test",
        "unittest",
        "matplotlib",
        "scipy",
        "PIL",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
    ],
    "semi_standalone": False,
    "site_packages": True,
    "arch": "arm64",  # Apple Silicon only
}

setup(
    app=APP,
    name="Dictate",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
