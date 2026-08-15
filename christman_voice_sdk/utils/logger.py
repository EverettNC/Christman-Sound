"""
Logging for the Christman voice SDK.

Structured logging with optional rich console output and file rotation.

WHAT CHANGED AND WHY
--------------------

1. `rich` WAS A HARD DEPENDENCY, AND THIS MODULE IS IMPORTED BY EVERYTHING.

       from rich.logging import RichHandler
       from rich.console import Console

   At module scope. `logger.py` is imported by tone_manager, tonescore_engine,
   tonescore_analyzer, emotion_embedder, audio_processor and more — so on a
   machine without `rich`, none of them import. The whole SDK is dead on a
   missing console-formatting library.

   `rich` is now optional. Without it the console handler is a plain
   StreamHandler and everything still runs.

2. THE WRAPPER DROPPED %-STYLE ARGUMENTS.

       def info(self, msg: str, **kwargs):
           self.logger.info(msg, **kwargs)

   No `*args`. The stdlib signature is `info(msg, *args, **kwargs)`, so:

       logger.info("Loaded %d items", 42)
       TypeError: Logger.info() takes 2 positional arguments but 3 were given

   Every module logging with %-style formatting — the correct way, since it
   defers formatting until the record is actually emitted — raised TypeError.
   `*args` is restored, and `stacklevel` is set so file:line points at the
   caller instead of at this wrapper.

3. A LOGGER WAS BUILT AT IMPORT TIME, WITH SIDE EFFECTS.

       logger = get_logger(name="christman_voice_sdk",
                           log_dir=Path.home() / ".christman_ai" / "logs", ...)

   Importing this module created directories in the user's home and opened a
   file handle. Import should not touch the filesystem. The module-level
   `logger` is now lazy — the handler is built on first use.

4. `handlers.clear()` WIPED SHARED STATE.

       self.logger.handlers.clear()

   `logging.getLogger(name)` returns a SHARED object. Two modules calling
   `get_logger("christman_voice_sdk")` meant the second wiped the first's
   handlers. Handlers are now added once and tracked with a marker attribute.

5. `propagate` was left True while a handler was attached, so records reached
   both this handler and the root logger's — every message twice if the app
   configured root logging.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _RICH = True
except ImportError:
    Console = RichHandler = None      # type: ignore[assignment]
    _RICH = False

#: Marker attribute so handlers are attached exactly once per named logger.
_CONFIGURED = "_christman_configured"

DEFAULT_LOG_DIR = Path(
    os.getenv("CHRISTMAN_LOG_DIR", Path.home() / ".christman_ai" / "logs")
)
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _resolve_level(level: Any) -> int:
    """
    Coerce a level to an int.

    `getattr(logging, level.upper())` raised AttributeError on a typo, with a
    message that did not mention logging levels.
    """
    if isinstance(level, int):
        return level
    name = str(level).upper()
    value = getattr(logging, name, None)
    if isinstance(value, int):
        return value
    logging.getLogger(__name__).error(
        "Unknown log level %r; falling back to INFO. Valid: "
        "DEBUG, INFO, WARNING, ERROR, CRITICAL.", level,
    )
    return logging.INFO


class Logger:
    """
    Thin wrapper over a stdlib logger.

    Every method forwards `*args` so %-style formatting works, and sets
    `stacklevel` so the emitted file:line is the CALLER, not this file.
    """

    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        level: Any = "INFO",
        file_logging: bool = True,
        propagate: bool = False,
        force_reconfigure: bool = False,
    ) -> None:
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(_resolve_level(level))
        self.logger.propagate = propagate
        self.file_logging_active = False

        if getattr(self.logger, _CONFIGURED, False) and not force_reconfigure:
            # Already set up. Do NOT clear handlers — this object is shared.
            self.file_logging_active = any(
                isinstance(h, RotatingFileHandler) for h in self.logger.handlers
            )
            return

        if force_reconfigure:
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)
                handler.close()

        self.logger.addHandler(self._console_handler())

        if file_logging:
            handler = self._file_handler(Path(log_dir) if log_dir else DEFAULT_LOG_DIR)
            if handler is not None:
                self.logger.addHandler(handler)
                self.file_logging_active = True

        setattr(self.logger, _CONFIGURED, True)

    @staticmethod
    def _console_handler() -> logging.Handler:
        """Rich when available, plain stderr otherwise."""
        if _RICH:
            handler: logging.Handler = RichHandler(
                console=Console(stderr=True),
                show_time=True, show_path=False,
                rich_tracebacks=True, tracebacks_show_locals=False,
            )
        else:
            handler = logging.StreamHandler(stream=sys.stderr)
            handler.setFormatter(logging.Formatter(FILE_FORMAT, "%Y-%m-%d %H:%M:%S"))
        handler.setLevel(logging.DEBUG)
        return handler

    def _file_handler(self, log_dir: Path) -> Optional[logging.Handler]:
        """
        Rotating file handler, or None when the directory is unwritable.

        Returns None rather than raising: losing file logs must not take the
        process down, but the console must say it happened.
        """
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                log_dir / f"{self.name}.log",
                maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8",
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(FILE_FORMAT, "%Y-%m-%d %H:%M:%S"))
            return handler
        except OSError as exc:
            logging.getLogger(__name__).error(
                "File logging disabled — cannot write %s: %s", log_dir, exc
            )
            return None

    # -- Forwarding. *args restored; stacklevel points at the caller. ---------

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stacklevel", 2)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stacklevel", 2)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stacklevel", 2)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stacklevel", 2)
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stacklevel", 2)
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stacklevel", 2)
        self.logger.exception(msg, *args, **kwargs)

    # -- Escape hatch ---------------------------------------------------------

    def addHandler(self, handler: logging.Handler) -> None:
        """Forwarded so callers treating this as a stdlib logger still work."""
        self.logger.addHandler(handler)

    def setLevel(self, level: Any) -> None:
        self.logger.setLevel(_resolve_level(level))

    @property
    def handlers(self) -> list:
        return self.logger.handlers

    def __repr__(self) -> str:
        return (
            f"<Logger {self.name!r} level={logging.getLevelName(self.logger.level)} "
            f"rich={_RICH} file={self.file_logging_active}>"
        )


def get_logger(name: str, **kwargs: Any) -> Logger:
    """Get or create a configured logger. Idempotent per name."""
    return Logger(name, **kwargs)


class _LazyLogger:
    """
    Module-level `logger`, built on first use.

    Importing this module previously created `~/.christman_ai/logs` and opened
    a file handle. Import should not touch the filesystem.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._real: Optional[Logger] = None

    def _get(self) -> Logger:
        if self._real is None:
            self._real = get_logger(
                self._name, log_dir=DEFAULT_LOG_DIR, level="INFO", file_logging=True
            )
        return self._real

    def __getattr__(self, item: str) -> Any:
        return getattr(self._get(), item)

    def __repr__(self) -> str:
        return (
            f"<LazyLogger {self._name!r} (not yet built)>"
            if self._real is None else repr(self._real)
        )


logger = _LazyLogger("christman_voice_sdk")

__all__ = ["logger", "get_logger", "Logger", "DEFAULT_LOG_DIR"]
