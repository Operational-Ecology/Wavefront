"""A small finwave-blue wave flourish for the terminal. Purely cosmetic.

Animates a travelling, foam-tipped wave in the finwave palette. No-ops on
non-TTY streams, under ``NO_COLOR``, or for dumb terminals, so it never
corrupts piped or logged output.
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import time
from typing import Optional

# finwave palette, deep water → crest (mirrors the logo's blue gradient)
_GRAD = [
    (14, 63, 133), (21, 88, 184), (31, 111, 230),
    (59, 143, 255), (93, 158, 255), (130, 185, 255),
]
_FOAM = (224, 238, 255)
_BLOCKS = " ▁▂▃▄▅▆▇█"


def supported(stream) -> bool:
    return (
        hasattr(stream, "isatty")
        and stream.isatty()
        and not os.environ.get("NO_COLOR")
        and not os.environ.get("WAVEFRONT_NO_ART")
        and os.environ.get("TERM", "") not in ("", "dumb")
    )


def _rgb(rgb) -> str:
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _surface(width: int, rows: int, phase: float) -> list[float]:
    """Water height in cells (0..rows) per column — two summed travelling waves."""
    out = []
    for x in range(width):
        h = 0.52 + 0.30 * math.sin(x * 0.26 - phase) + 0.14 * math.sin(x * 0.11 + phase * 0.6)
        out.append(max(0.0, min(1.0, h)) * rows)
    return out


def _render_frame(width: int, rows: int, phase: float) -> str:
    surf = _surface(width, rows, phase)
    lines = []
    for r in range(rows):                       # r = 0 is the top row
        depth_from_surface = r                   # 0 near crest → lighter
        line = []
        for x in range(width):
            band = surf[x] - (rows - 1 - r)      # cells of water in this row (>1 = full)
            if band <= 0:
                line.append(" ")
                continue
            crest = band < 1.0                   # the topmost filled cell
            block = _BLOCKS[min(8, max(1, int(round(band * 8))))] if crest else "█"
            if crest:
                color = _FOAM
            else:
                gi = min(len(_GRAD) - 1, depth_from_surface)
                color = _GRAD[len(_GRAD) - 1 - gi] if False else _GRAD[gi]
            line.append(_rgb(color) + block)
        lines.append("".join(line) + "\x1b[0m")
    return "\n".join(lines)


def wave(stream=None, *, duration: float = 1.3, fps: int = 30,
         width: Optional[int] = None, rows: int = 4) -> None:
    """Play the wave flourish, then leave the terminal clean."""
    stream = stream or sys.stderr
    if not supported(stream):
        return
    cols = shutil.get_terminal_size((80, 24)).columns
    w = min(width or cols - 2, 60)
    frames = max(1, int(duration * fps))
    stream.write("\x1b[?25l")  # hide cursor
    try:
        for f in range(frames):
            stream.write(_render_frame(w, rows, f / fps * 6.5))
            if f < frames - 1:
                stream.write(f"\x1b[{rows - 1}A\r")  # back to top of the wave
            stream.flush()
            time.sleep(1.0 / fps)
        stream.write("\n")
    finally:
        stream.write("\x1b[?25h\x1b[0m")  # restore cursor + reset
        stream.flush()


def banner(stream=None) -> None:
    """A one-shot wave + wordmark, used by the bare ``wavefront`` command."""
    from . import __version__
    stream = stream or sys.stderr
    wave(stream, duration=1.1)
    if supported(stream):
        stream.write(_rgb(_GRAD[3]) + "  wavefront " + _rgb(_GRAD[5])
                     + f"v{__version__}\x1b[0m  " + "\x1b[2mfinwave datasets, one call\x1b[0m\n")
    else:
        stream.write(f"wavefront v{__version__} — finwave datasets, one call\n")


if __name__ == "__main__":  # quick visual check: `python -m wavefront._art`
    wave(sys.stdout, duration=3.0)
