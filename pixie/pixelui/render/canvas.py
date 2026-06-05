"""RGBA pixel canvas with compositing and PNG export."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

Color = tuple[int, int, int, int]

BLACK: Color = (0, 0, 0, 255)
WHITE: Color = (255, 255, 255, 255)

_SKY_BLUE: Color = (135, 206, 235, 255)


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass
class Canvas:
    width: int
    height: int
    _data: bytearray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        object.__setattr__(self, "_data", bytearray(self.width * self.height * 4))
        self.clear(WHITE)

    def _offset(self, x: int, y: int) -> int:
        return (y * self.width + x) * 4

    def clear(self, color: Color = WHITE) -> None:
        r, g, b, a = color
        pixel = bytes([r, g, b, a])
        row_len = self.width * 4
        for y in range(self.height):
            off = y * row_len
            self._data[off : off + row_len] = pixel * self.width

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        off = self._offset(x, y)
        r, g, b, a = color
        self._data[off] = r
        self._data[off + 1] = g
        self._data[off + 2] = b
        self._data[off + 3] = a

    def get_pixel(self, x: int, y: int) -> Color:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return (0, 0, 0, 0)
        off = self._offset(x, y)
        return (
            self._data[off],
            self._data[off + 1],
            self._data[off + 2],
            self._data[off + 3],
        )

    def blit(
        self,
        source: Canvas,
        src_x: int,
        src_y: int,
        dst_x: int,
        dst_y: int,
        width: int,
        height: int,
    ) -> None:
        if width <= 0 or height <= 0:
            return
        for dy in range(height):
            sy = src_y + dy
            if not (0 <= sy < source.height):
                continue
            for dx in range(width):
                sx = src_x + dx
                if not (0 <= sx < source.width):
                    break
                self.set_pixel(dst_x + dx, dst_y + dy, source.get_pixel(sx, sy))

    def blit_from_bytes(
        self,
        data: bytes | bytearray,
        src_width: int,
        src_height: int,
        src_x: int,
        src_y: int,
        dst_x: int,
        dst_y: int,
        width: int,
        height: int,
        fg_color: Color = BLACK,
        bg_color: Color = WHITE,
    ) -> None:
        """Blit from a 1-bit mask, mapping 1-bits → *fg_color*, 0-bits → *bg_color*."""
        if width <= 0 or height <= 0:
            return
        src_stride = (src_width + 7) // 8
        for dy in range(height):
            sy = src_y + dy
            if not (0 <= sy < src_height):
                continue
            for w in range(width):
                sx = src_x + w
                if not (0 <= sx < src_width):
                    break
                byte_idx = sx // 8
                bit_idx = 7 - (sx % 8)
                bit = bool(data[sy * src_stride + byte_idx] & (1 << bit_idx))
                self.set_pixel(dst_x + w, dst_y + dy, fg_color if bit else bg_color)

    def to_png(self, filepath: str | None = None) -> bytes:
        """Encode the canvas as an RGBA PNG and optionally write to *filepath*."""
        ihdr_data = struct.pack(">IIBBBBB", self.width, self.height, 8, 6, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF

        raw_rows = bytearray()
        stride = self.width * 4
        for y in range(self.height):
            raw_rows.append(0)
            raw_rows.extend(self._data[y * stride : (y + 1) * stride])

        compressed = zlib.compress(bytes(raw_rows))
        idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF

        png = bytearray()
        png += b"\x89PNG\r\n\x1a\n"
        png += struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
        png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

        data = bytes(png)
        if filepath:
            with open(filepath, "wb") as f:
                f.write(data)
        return data

    def __getitem__(self, pos: tuple[int, int]) -> Color:
        return self.get_pixel(*pos)

    def __setitem__(self, pos: tuple[int, int], color: Color) -> None:
        self.set_pixel(pos[0], pos[1], color)

    def copy(self) -> Canvas:
        new = Canvas(self.width, self.height)
        new._data[:] = self._data[:]
        return new

    def __repr__(self) -> str:
        return f"Canvas({self.width}x{self.height})"
