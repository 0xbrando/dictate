# Dependency Licenses

All dependencies used by Dictate and their respective licenses.

## Python Packages

| Package | Version | License | URL |
|---------|---------|---------|-----|
| mlx | >=0.5.0 | MIT | https://github.com/ml-explore/mlx |
| mlx-whisper | >=0.1.0 | MIT | https://github.com/ml-explore/mlx-examples |
| mlx-lm | >=0.5.0 | MIT | https://github.com/ml-explore/mlx-examples |
| sounddevice | >=0.4.6 | MIT | https://github.com/spatialaudio/python-sounddevice |
| scipy | >=1.10.0 | BSD-3-Clause | https://github.com/scipy/scipy |
| numpy | >=1.24.0 | BSD-3-Clause | https://github.com/numpy/numpy |
| pyperclip | >=1.8.2 | BSD-3-Clause | https://github.com/asweigart/pyperclip |
| pynput | >=1.7.6 | LGPLv3 | https://github.com/moses-palmer/pynput |
| rumps | >=0.4.0 | BSD-3-Clause | https://github.com/jaredks/rumps |

### Development

| Package | Version | License | URL |
|---------|---------|---------|-----|
| pytest | >=7.0 | MIT | https://github.com/pytest-dev/pytest |
| mypy | >=1.0 | MIT | https://github.com/python/mypy |
| ruff | >=0.1.0 | MIT | https://github.com/astral-sh/ruff |

## ML Models

| Model | License | URL |
|-------|---------|-----|
| whisper-large-v3-turbo | MIT | https://huggingface.co/mlx-community/whisper-large-v3-turbo |
| Qwen2.5-3B-Instruct-4bit | Qwen Research | https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit |
| Qwen2.5-7B-Instruct-4bit | Qwen Research | https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit |
| Qwen2.5-14B-Instruct-4bit | Qwen Research | https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit |

## License Compatibility

All dependencies are compatible with the MIT license of this project.

**Note:** pynput is licensed under LGPLv3, which is compatible with MIT when used as a library (dynamically linked). No modifications to pynput source code are distributed with this project.
