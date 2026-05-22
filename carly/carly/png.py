import struct
import zlib


def _crc32(chunk_type: bytes, data: bytes) -> int:
    c = zlib.crc32(chunk_type)
    c = zlib.crc32(data, c)
    return c


def _chunk(out, chunk_type: bytes, data: bytes) -> None:
    out.write(struct.pack(">I", len(data)))
    out.write(chunk_type)
    out.write(data)
    out.write(struct.pack(">I", _crc32(chunk_type, data) & 0xFFFFFFFF))


_HEADER = b"\x89PNG\r\n\x1a\n"


def dump_png(out, pixels: list[list[tuple[int, int, int]]]) -> None:
    h = len(pixels)
    w = len(pixels[0]) if h else 0
    out.write(_HEADER)

    ihdr = struct.pack(">2I5B", w, h, 8, 2, 0, 0, 0)
    _chunk(out, b"IHDR", ihdr)

    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))
    compressed = zlib.compress(raw)
    _chunk(out, b"IDAT", compressed)
    _chunk(out, b"IEND", b"")


def save_png(pixels: list[list[tuple[int, int, int]]], path: str) -> None:
    with open(path, "wb") as f:
        dump_png(f, pixels)
