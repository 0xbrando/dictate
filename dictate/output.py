from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pyperclip
from pynput.keyboard import Controller as KeyboardController

if TYPE_CHECKING:
    from dictate.config import OutputMode

logger = logging.getLogger(__name__)

TYPING_DELAY_SECONDS = 0.05


class OutputHandler(ABC):
    @abstractmethod
    def output(self, text: str) -> None:
        ...


class ClipboardOutput(OutputHandler):
    def output(self, text: str) -> None:
        pyperclip.copy(text)


class TyperOutput(OutputHandler):
    def __init__(self) -> None:
        self._controller = KeyboardController()

    def output(self, text: str) -> None:
        try:
            from pynput.keyboard import Key

            pyperclip.copy(text)
            time.sleep(TYPING_DELAY_SECONDS)
            self._controller.press(Key.cmd)
            self._controller.press('v')
            self._controller.release('v')
            self._controller.release(Key.cmd)
        except pyperclip.PyperclipException as e:
            logger.error("Clipboard not available: %s", e)
            # Try to fall back to direct typing
            try:
                self._controller.type(text)
            except Exception as type_e:
                logger.error("Failed to type text directly: %s", type_e)
                raise RuntimeError(f"Failed to output text (clipboard unavailable and typing failed): {type_e}") from type_e
        except Exception as e:
            logger.error("Failed to paste text: %s", e)
            raise


MAX_AGGREGATED_CHARS = 50_000


class TextAggregator:
    def __init__(self) -> None:
        self._full_text = ""

    @property
    def full_text(self) -> str:
        return self._full_text

    def append(self, text: str) -> str:
        text = text.strip()
        if self._full_text:
            self._full_text = self._full_text.rstrip() + "\n" + text
        else:
            self._full_text = text
        if len(self._full_text) > MAX_AGGREGATED_CHARS:
            self._full_text = self._full_text[-MAX_AGGREGATED_CHARS:]
        return self._full_text

    def clear(self) -> None:
        self._full_text = ""


def create_output_handler(mode: "OutputMode") -> OutputHandler:
    from dictate.config import OutputMode

    if mode == OutputMode.CLIPBOARD:
        return ClipboardOutput()

    # TyperOutput copies to clipboard AND pastes
    return TyperOutput()
