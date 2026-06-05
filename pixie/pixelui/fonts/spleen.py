"""Load glyph data from the vendored Spleen 8x16 spritesheet."""

from __future__ import annotations

import os
import struct
import zlib

from pixelui.fonts.sprite_font import Glyph

_SPLEEN_PATH = os.path.join(os.path.dirname(__file__), "spleen-8x16.png")


def _decompress_png(path: str) -> tuple[bytes, int, int]:
    """Read a 1-bit PNG and return (raw_pixels, width, height)."""
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a valid PNG")

    pos = 8
    chunks: dict[bytes, bytes] = {}
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        chunk_type = raw[pos + 4 : pos + 8]
        chunks[chunk_type] = raw[pos + 8 : pos + 8 + length]
        pos += 12 + length

    ihdr = chunks[b"IHDR"]
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", ihdr[:13])

    if bit_depth != 1 or color_type != 3:
        raise ValueError(f"Expected 1-bit colormap PNG, got bit_depth={bit_depth} color_type={color_type}")

    idat = chunks[b"IDAT"]
    decompressed = zlib.decompress(idat)
    stride = (width + 7) // 8

    # Apply PNG filters to reconstruct raw scanlines
    scanlines = bytearray()
    sp = 0
    for _ in range(height):
        filter_type = decompressed[sp]
        sp += 1
        row = bytearray(decompressed[sp : sp + stride])
        sp += stride

        if filter_type == 0:
            pass
        elif filter_type == 1:
            for i in range(stride - 1, -1, -1):
                left = row[i - 1] if i > 0 else 0
                row[i] = (row[i] + left) & 0xFF
        elif filter_type == 2:
            prev = scanlines[-stride:] if scanlines else bytes(stride)
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif filter_type == 3:
            prev = scanlines[-stride:] if scanlines else bytes(stride)
            for i in range(stride):
                left = row[i - 1] if i > 0 else 0
                up = prev[i]
                row[i] = (row[i] + (left + up) // 2) & 0xFF
        elif filter_type == 4:
            prev = scanlines[-stride:] if scanlines else bytes(stride)
            for i in range(stride):
                a = row[i - 1] if i > 0 else 0
                b = prev[i]
                c = prev[i - 1] if i > 0 else 0
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                row[i] = (row[i] + pr) & 0xFF

        scanlines.extend(row)

    return bytes(scanlines), width, height


def load_spleen_glyphs() -> list[Glyph]:
    """Read the vendored Spleen 8x16 spritesheet and extract glyphs.

    Returns Glyph objects for ASCII printable characters (0x20-0x7E).
    """
    pixels, width, height = _decompress_png(_SPLEEN_PATH)
    stride = (width + 7) // 8
    chars_per_row = width // 8

    glyphs: list[Glyph] = []
    for gidx in range(chars_per_row):
        char = chr(0x20 + gidx)
        pixel_rows: list[int] = []
        for row in range(height):
            byte_val = pixels[row * stride + gidx]
            # Spleen palette: index 0 = black (background), index 1 = gray (ink).
            # The 1-bit value maps directly: 1 = black (ink), 0 = white (paper).
            pixel_rows.append(byte_val)
        glyphs.append(Glyph(char=char, pixels=pixel_rows))

    return glyphs
