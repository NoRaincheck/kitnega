"""Layout engine - text wrapping, multi-line rendering, element composition."""

from __future__ import annotations

from dataclasses import dataclass

from pixelui.fonts.sprite_font import SpriteFont


@dataclass(frozen=True)
class LayoutLine:
    """A single rendered line of text or an element."""

    kind: str  # "text" | "element" | "blank"
    content: str  # text string, element name, or ""
    x: int  # horizontal position (pixels from canvas left)
    y: int  # vertical position (pixels from canvas top)


@dataclass(frozen=True)
class LayoutBlock:
    """A block of layout instructions."""

    lines: list[str] = None  # text lines to render
    elements: list[tuple[str, int, int]] = None  # (name, x, y) pairs
    width: int | None = None  # max character width for wrapping


@dataclass
class LayoutResult:
    """The output of a layout pass."""

    lines: list[LayoutLine] = None  # all positioned items
    width: int = 0  # total canvas width needed
    height: int = 0  # total canvas height needed


def wrap_text(text: str, max_chars: int) -> list[str]:
    """Word-wrap text to fit within max_chars columns.

    Preserves newlines and breaks words that exceed the limit.
    """
    if max_chars <= 0:
        return []

    result: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            result.append("")
            continue

        while len(paragraph) > max_chars:
            break_at = max_chars
            for i in range(max_chars - 1, -1, -1):
                if paragraph[i] == " ":
                    break_at = i
                    break

            result.append(paragraph[:break_at].rstrip())
            paragraph = paragraph[break_at:].lstrip()

        if paragraph:
            result.append(paragraph)

    return result


def layout_text(
    text: str,
    font: SpriteFont,
    *,
    x_start: int = 0,
    y_start: int = 0,
    max_width_chars: int | None = None,
) -> LayoutResult:
    """Lay out plain text and return positioned lines."""
    if max_width_chars is not None:
        lines = wrap_text(text, max_width_chars)
    else:
        lines = text.split("\n")

    result_lines: list[LayoutLine] = []
    y = y_start
    total_width = 0

    for line in lines:
        x = x_start
        if not line:
            y += font.glyph_height
            continue

        for ch in line:
            glyph = font.get_glyph(ch)
            if glyph is None:
                x += font.glyph_width
                continue
            result_lines.append(LayoutLine(kind="text", content=ch, x=x, y=y))
            x += font.glyph_width

        line_width = len(line) * font.glyph_width
        if line_width > total_width:
            total_width = line_width

        y += font.glyph_height

    return LayoutResult(
        lines=result_lines,
        width=total_width + x_start,
        height=y - y_start if y > y_start else font.glyph_height,
    )


def layout_panel(
    title: str | None,
    body: str,
    border_chars: dict[str, str] | None = None,
    *,
    font: SpriteFont,
    x: int = 0,
    y: int = 0,
) -> LayoutResult:
    """Lay out a bordered panel with an optional title."""
    border = border_chars or {
        "tl": "\u250c",
        "tr": "\u2510",
        "bl": "\u2514",
        "br": "\u2518",
        "h": "\u2500",
        "v": "\u2502",
    }

    max_body_chars = 40
    wrapped_lines = wrap_text(body, max_body_chars) if body else []
    panel_width = max_body_chars + 2

    result_lines: list[LayoutLine] = []

    # Top border row
    for i in range(panel_width):
        if i == 0:
            ch = "tl"
        elif i == panel_width - 1:
            ch = "tr"
        else:
            ch = "h"
        content = border.get(ch, str(i)) if isinstance(border, dict) else ch
        result_lines.append(LayoutLine(kind="text", content=content[0], x=x + i * font.glyph_width, y=y))

    # Title row (if any)
    title_y = y + font.glyph_height
    if title:
        result_lines.append(LayoutLine(kind="text", content=border.get("v", "|")[0], x=x, y=title_y))
        start_x = x + font.glyph_width
        for i, ch in enumerate(title):
            glyph = font.get_glyph(ch)
            if glyph:
                result_lines.append(LayoutLine(kind="text", content=ch, x=start_x + i * font.glyph_width, y=title_y))
        result_lines.append(
            LayoutLine(
                kind="text", content=border.get("v", "|")[0], x=x + (panel_width - 1) * font.glyph_width, y=title_y
            )
        )

    # Body rows
    for row_idx, line in enumerate(wrapped_lines):
        row_y = title_y + font.glyph_height if title else title_y
        result_lines.append(LayoutLine(kind="text", content=border.get("v", "|")[0], x=x, y=row_y))

        body_x = x + font.glyph_width
        for ch in line:
            glyph = font.get_glyph(ch)
            if glyph:
                result_lines.append(LayoutLine(kind="text", content=ch, x=body_x, y=row_y))
                body_x += font.glyph_width

        result_lines.append(
            LayoutLine(
                kind="text", content=border.get("v", "|")[0], x=x + (panel_width - 1) * font.glyph_width, y=row_y
            )
        )

    # Bottom border row
    bot_y = title_y + len(wrapped_lines) * font.glyph_height
    for i in range(panel_width):
        if i == 0:
            ch = "bl"
        elif i == panel_width - 1:
            ch = "br"
        else:
            ch = "h"
        content = border.get(ch, str(i)) if isinstance(border, dict) else ch
        result_lines.append(LayoutLine(kind="text", content=content[0], x=x + i * font.glyph_width, y=bot_y))

    return LayoutResult(
        lines=result_lines,
        width=panel_width * font.glyph_width,
        height=(bot_y - y) + font.glyph_height,
    )


def layout_bordered_box(
    text: str,
    font: SpriteFont,
    *,
    x: int = 0,
    y: int = 0,
    border: str = "table",
) -> LayoutResult:
    """Lay out a tight bordered box around a single line of text.

    The border is drawn using 8×8 element tiles, tightly wrapped around
    the 8×16 character glyphs with no extra padding.
    """
    char_count = len(text)
    content_w = char_count * font.glyph_width
    content_h = font.glyph_height

    bw = 8  # border tile width (pixels)
    bh = 8  # border tile height (pixels)

    total_w = bw + content_w + bw
    total_h = bh + content_h + bh

    result_lines: list[LayoutLine] = []

    bl = f"{border}-tl"
    bt = f"{border}-top"
    br = f"{border}-tr"
    blf = f"{border}-left"
    brf = f"{border}-right"
    bb = f"{border}-bottom"
    bbl = f"{border}-bl"
    bbr = f"{border}-br"

    # Top border row (y = y)
    result_lines.append(LayoutLine(kind="element", content=bl, x=x, y=y))
    for i in range(char_count):
        result_lines.append(LayoutLine(kind="element", content=bt, x=x + bw + i * font.glyph_width, y=y))
    result_lines.append(LayoutLine(kind="element", content=br, x=x + bw + content_w, y=y))

    # Content row — top half (y = y + bh)
    cx = y + bh
    result_lines.append(LayoutLine(kind="element", content=blf, x=x, y=cx))
    for i, ch in enumerate(text):
        result_lines.append(LayoutLine(kind="text", content=ch, x=x + bw + i * font.glyph_width, y=cx))
    result_lines.append(LayoutLine(kind="element", content=brf, x=x + bw + content_w, y=cx))

    # Content row — bottom half (y = y + bh + font.glyph_height // 2)
    # The 8×16 glyphs from the row above already occupy this vertical band.
    # Only the border edges need to be continued here.
    cy = y + bh + content_h // 2
    result_lines.append(LayoutLine(kind="element", content=blf, x=x, y=cy))
    result_lines.append(LayoutLine(kind="element", content=brf, x=x + bw + content_w, y=cy))

    # Bottom border row (y = y + bh + content_h)
    by = y + bh + content_h
    result_lines.append(LayoutLine(kind="element", content=bbl, x=x, y=by))
    for i in range(char_count):
        result_lines.append(LayoutLine(kind="element", content=bb, x=x + bw + i * font.glyph_width, y=by))
    result_lines.append(LayoutLine(kind="element", content=bbr, x=x + bw + content_w, y=by))

    return LayoutResult(lines=result_lines, width=total_w, height=total_h)
