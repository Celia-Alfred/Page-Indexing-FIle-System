
import io

import pymupdf


FULL_WIDTH_RATIO = 0.72
MIN_BLOCKS_FOR_COLUMNS = 4
MIN_COLUMN_GAP_RATIO = 0.12
MIN_SIDE_BLOCKS = 2


def reconstruct_line_text(spans):
    """Reconstruct text from spans in horizontal reading order."""
    spans = sorted(
        spans,
        key=lambda span: span.get("bbox", [0, 0, 0, 0])[0],
    )

    parts = []
    previous_right = None
    previous_size = None

    for span in spans:
        text = span.get("text", "")
        bbox = span.get("bbox", [0, 0, 0, 0])
        x0, _, x1, _ = bbox
        size = span.get("size", 0)

        if (
            parts
            and previous_right is not None
            and x0 - previous_right
            > max(size, previous_size or 0, 1) * 0.15
            and not text.startswith((" ", ",", ".", ";", ":", ")", "]"))
            and not parts[-1].endswith((" ", "-", "/", "(", "["))
        ):
            parts.append(" ")

        parts.append(text)
        previous_right = x1
        previous_size = size

    return "".join(parts).strip()


def _bbox(block):
    return block.get("bbox", [0, 0, 0, 0])


def _text(block):
    return block.get("text", "").strip()


def _is_full_width(block, page_width):
    x0, _, x1, _ = _bbox(block)
    return (x1 - x0) >= page_width * FULL_WIDTH_RATIO


def _valid_blocks(blocks):
    return [
        block
        for block in blocks
        if _text(block) and len(_bbox(block)) >= 4
    ]


def _sort_vertical(blocks):
    """Stable top-to-bottom ordering."""
    return sorted(
        enumerate(blocks),
        key=lambda item: (
            _bbox(item[1])[1],
            _bbox(item[1])[0],
            item[0],
        ),
    )


def _detect_columns(blocks, page_width):
    """
    Detect two columns conservatively.

    Return a divider only when blocks are clearly separated into left
    and right columns with sufficient vertical overlap.
    """
    candidates = [
        block
        for block in _valid_blocks(blocks)
        if not _is_full_width(block, page_width)
    ]

    if len(candidates) < MIN_BLOCKS_FOR_COLUMNS:
        return None

    centers = sorted(
        ((_bbox(block)[0] + _bbox(block)[2]) / 2)
        for block in candidates
    )

    gaps = [
        (centers[i + 1] - centers[i], i)
        for i in range(len(centers) - 1)
    ]

    if not gaps:
        return None

    largest_gap, split_index = max(gaps)

    if largest_gap < page_width * MIN_COLUMN_GAP_RATIO:
        return None

    divider = (centers[split_index] + centers[split_index + 1]) / 2

    left = [
        block for block in candidates
        if (_bbox(block)[0] + _bbox(block)[2]) / 2 < divider
    ]
    right = [
        block for block in candidates
        if (_bbox(block)[0] + _bbox(block)[2]) / 2 >= divider
    ]

    if len(left) < MIN_SIDE_BLOCKS or len(right) < MIN_SIDE_BLOCKS:
        return None

    if any(_bbox(block)[2] > divider for block in left):
        return None

    if any(_bbox(block)[0] < divider for block in right):
        return None

    left_top = min(_bbox(block)[1] for block in left)
    left_bottom = max(_bbox(block)[3] for block in left)
    right_top = min(_bbox(block)[1] for block in right)
    right_bottom = max(_bbox(block)[3] for block in right)

    vertical_overlap = min(left_bottom, right_bottom) - max(
        left_top, right_top
    )

    if vertical_overlap <= 0:
        return None

    return divider


def _order_region(blocks, page_width):
    """
    Order a region as left column then right column when confidently
    detected. Otherwise retain top-to-bottom order.
    """
    if len(blocks) < MIN_BLOCKS_FOR_COLUMNS:
        return [block for _, block in _sort_vertical(blocks)]

    divider = _detect_columns(blocks, page_width)

    if divider is None:
        return [block for _, block in _sort_vertical(blocks)]

    left = []
    right = []

    for block in blocks:
        x0, _, x1, _ = _bbox(block)

        if x1 <= divider:
            left.append(block)
        elif x0 >= divider:
            right.append(block)
        else:
            return [block for _, block in _sort_vertical(blocks)]

    if len(left) < MIN_SIDE_BLOCKS or len(right) < MIN_SIDE_BLOCKS:
        return [block for _, block in _sort_vertical(blocks)]

    return (
        [block for _, block in _sort_vertical(left)]
        + [block for _, block in _sort_vertical(right)]
    )


def _group_vertical_bands(blocks):
    """
    Group vertically overlapping blocks into layout bands.

    Blocks in separate vertical bands are not used as evidence for
    each other's column structure.
    """
    ordered = [
        block for _, block in _sort_vertical(blocks)
    ]

    bands = []
    current_band = []
    band_bottom = float("-inf")

    for block in ordered:
        y0, y1 = _bbox(block)[1], _bbox(block)[3]

        if current_band and y0 >= band_bottom:
            bands.append(current_band)
            current_band = []
            band_bottom = float("-inf")

        current_band.append(block)
        band_bottom = max(band_bottom, y1)

    if current_band:
        bands.append(current_band)

    return bands


def _vertical_overlap(a, b):
    return max(
        0.0,
        min(_bbox(a)[3], _bbox(b)[3])
        - max(_bbox(a)[1], _bbox(b)[1]),
    )


def _sort_blocks_by_layout(blocks, page_width):
    """
    Sort page blocks while respecting full-width separators and columns.

    Full-width blocks separate layout regions. Each region is processed
    independently, reducing the chance that columns from different
    sections are mixed together.
    """
    blocks = _valid_blocks(blocks)

    if len(blocks) < 2:
        return blocks

    separators = [
        block for block in blocks
        if _is_full_width(block, page_width)
    ]

    if not separators:
        output = []
        for band in _group_vertical_bands(blocks):
            output.extend(_order_region(band, page_width))
        return output

    separators = [
        block for _, block in _sort_vertical(separators)
    ]

    output = []
    remaining = list(blocks)

    for separator in separators:
        sep_y0, sep_y1 = _bbox(separator)[1], _bbox(separator)[3]

        overlapping = [
            block for block in remaining
            if block is not separator
            and _vertical_overlap(block, separator) > 0
        ]

        if overlapping:
            return blocks

        above = [
            block for block in remaining
            if block is not separator
            and _bbox(block)[3] <= sep_y0
        ]

        for band in _group_vertical_bands(above):
            output.extend(_order_region(band, page_width))

        output.append(separator)

        removed_ids = {id(block) for block in above}
        removed_ids.add(id(separator))

        remaining = [
            block for block in remaining
            if id(block) not in removed_ids
        ]

    for band in _group_vertical_bands(remaining):
        output.extend(_order_region(band, page_width))

    return output


def _extract_ordered_blocks(page, page_data):
    """Extract text blocks with line, font, and bounding-box metadata."""
    raw_blocks = [
        block
        for block in page_data.get("blocks", [])
        if block.get("type", 0) == 0
    ]

    extracted = []

    for block in raw_blocks:
        lines = []

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = reconstruct_line_text(spans)

            if not text:
                continue

            lines.append({
                "text": text,
                "bbox": line.get("bbox", []),
                "spans": [
                    {
                        "text": span.get("text", ""),
                        "font": span.get("font", ""),
                        "font_size": span.get("size", 0),
                        "flags": span.get("flags", 0),
                        "bbox": span.get("bbox", []),
                    }
                    for span in spans
                ],
            })

        block_text = "\n".join(
            line["text"] for line in lines
        ).strip()

        if block_text:
            extracted.append({
                "bbox": block.get("bbox", [0, 0, 0, 0]),
                "lines": lines,
                "text": block_text,
            })

    return _sort_blocks_by_layout(extracted, page.rect.width)


def extract_pages(pdf_bytes):
    """Extract text and layout metadata from each PDF page."""
    pages = []

    with pymupdf.open(
        stream=io.BytesIO(pdf_bytes),
        filetype="pdf",
    ) as document:
        for index, page in enumerate(document):
            page_data = page.get_text("dict", sort=False)
            blocks = _extract_ordered_blocks(page, page_data)

            pages.append({
                "page_number": index + 1,
                "text": "\n\n".join(
                    block["text"] for block in blocks
                ).strip(),
                "blocks": blocks,
            })

    return pages


def extract_pdf_toc(pdf_bytes):
    """Extract embedded PDF bookmarks, if available."""
    with pymupdf.open(
        stream=io.BytesIO(pdf_bytes),
        filetype="pdf",
    ) as document:
        return document.get_toc()
