
import io
from pathlib import Path
from tempfile import NamedTemporaryFile

import pymupdf

from services.ocr_service import extract_text_from_image
MIN_NATIVE_TEXT_LENGTH = 30
RENDER_SCALE = 2

FULL_WIDTH_RATIO = 0.72
MIN_BLOCKS_FOR_COLUMNS = 4
MIN_COLUMN_GAP_RATIO = 0.12
MIN_SIDE_BLOCKS = 2


def _to_float_bbox(bbox):
    """Convert a bounding box into JSON-safe float coordinates."""

    if not bbox or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]

    return [float(value) for value in bbox[:4]]


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
            and not text.startswith(
                (" ", ",", ".", ";", ":", ")", "]")
            )
            and not parts[-1].endswith(
                (" ", "-", "/", "(", "[")
            )
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

    Return a divider only when blocks are clearly separated
    into left and right columns.
    """

    candidates = [
        block
        for block in _valid_blocks(blocks)
        if not _is_full_width(block, page_width)
    ]

    if len(candidates) < MIN_BLOCKS_FOR_COLUMNS:
        return None

    centers = sorted(
        (
            (_bbox(block)[0] + _bbox(block)[2]) / 2
        )
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

    divider = (
        centers[split_index] + centers[split_index + 1]
    ) / 2

    left = [
        block
        for block in candidates
        if (_bbox(block)[0] + _bbox(block)[2]) / 2 < divider
    ]

    right = [
        block
        for block in candidates
        if (_bbox(block)[0] + _bbox(block)[2]) / 2 >= divider
    ]

    if (
        len(left) < MIN_SIDE_BLOCKS
        or len(right) < MIN_SIDE_BLOCKS
    ):
        return None

    if any(_bbox(block)[2] > divider for block in left):
        return None

    if any(_bbox(block)[0] < divider for block in right):
        return None

    left_top = min(_bbox(block)[1] for block in left)
    left_bottom = max(_bbox(block)[3] for block in left)

    right_top = min(_bbox(block)[1] for block in right)
    right_bottom = max(_bbox(block)[3] for block in right)

    vertical_overlap = min(
        left_bottom,
        right_bottom,
    ) - max(
        left_top,
        right_top,
    )

    if vertical_overlap <= 0:
        return None

    return divider


def _order_region(blocks, page_width):
    """Order a region as left column then right column when detected."""

    if len(blocks) < MIN_BLOCKS_FOR_COLUMNS:
        return [
            block
            for _, block in _sort_vertical(blocks)
        ]

    divider = _detect_columns(blocks, page_width)

    if divider is None:
        return [
            block
            for _, block in _sort_vertical(blocks)
        ]

    left = []
    right = []

    for block in blocks:

        x0, _, x1, _ = _bbox(block)

        if x1 <= divider:
            left.append(block)

        elif x0 >= divider:
            right.append(block)

        else:
            return [
                block
                for _, block in _sort_vertical(blocks)
            ]

    if (
        len(left) < MIN_SIDE_BLOCKS
        or len(right) < MIN_SIDE_BLOCKS
    ):
        return [
            block
            for _, block in _sort_vertical(blocks)
        ]

    return (
        [
            block
            for _, block in _sort_vertical(left)
        ]
        +
        [
            block
            for _, block in _sort_vertical(right)
        ]
    )


def _group_vertical_bands(blocks):
    """Group vertically overlapping blocks into layout bands."""

    ordered = [
        block
        for _, block in _sort_vertical(blocks)
    ]

    bands = []
    current_band = []
    band_bottom = float("-inf")

    for block in ordered:

        y0 = _bbox(block)[1]
        y1 = _bbox(block)[3]

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
    """Sort page blocks while respecting full-width separators and columns."""

    blocks = _valid_blocks(blocks)

    if len(blocks) < 2:
        return blocks

    separators = [
        block
        for block in blocks
        if _is_full_width(block, page_width)
    ]

    if not separators:

        output = []

        for band in _group_vertical_bands(blocks):
            output.extend(
                _order_region(band, page_width)
            )

        return output

    separators = [
        block
        for _, block in _sort_vertical(separators)
    ]

    output = []
    remaining = list(blocks)

    for separator in separators:

        sep_y0 = _bbox(separator)[1]
        sep_y1 = _bbox(separator)[3]

        overlapping = [
            block
            for block in remaining
            if block is not separator
            and _vertical_overlap(block, separator) > 0
        ]

        if overlapping:
            return blocks

        above = [
            block
            for block in remaining
            if block is not separator
            and _bbox(block)[3] <= sep_y0
        ]

        for band in _group_vertical_bands(above):
            output.extend(
                _order_region(band, page_width)
            )

        output.append(separator)

        removed_ids = {
            id(block)
            for block in above
        }

        removed_ids.add(id(separator))

        remaining = [
            block
            for block in remaining
            if id(block) not in removed_ids
        ]

    for band in _group_vertical_bands(remaining):
        output.extend(
            _order_region(band, page_width)
        )

    return output


def _extract_ordered_blocks(page, page_data):
    """
    Extract native text blocks with stable IDs, line metadata,
    font metadata, and bounding boxes.
    """

    raw_blocks = [
        (raw_index, block)
        for raw_index, block in enumerate(
            page_data.get("blocks", [])
        )
        if block.get("type", 0) == 0
    ]

    extracted = []

    for raw_index, block in raw_blocks:

        block_no = int(
            block.get("number", raw_index)
        )

        block_id = (
            f"page-{page.number + 1}-block-{block_no}"
        )

        lines = []

        for line in block.get("lines", []):

            spans = line.get("spans", [])
            text = reconstruct_line_text(spans)

            if not text:
                continue

            lines.append({
                "text": text,
                "bbox": _to_float_bbox(
                    line.get("bbox", [])
                ),
                "spans": [
                    {
                        "text": span.get("text", ""),
                        "font": span.get("font", ""),
                        "font_size": float(
                            span.get("size", 0)
                        ),
                        "flags": int(
                            span.get("flags", 0)
                        ),
                        "bbox": _to_float_bbox(
                            span.get("bbox", [])
                        ),
                    }
                    for span in spans
                ],
            })

        block_text = "\n".join(
            line["text"]
            for line in lines
        ).strip()

        if block_text:

            extracted.append({
                "block_id": block_id,
                "block_no": block_no,
                "bbox": _to_float_bbox(
                    block.get("bbox", [0, 0, 0, 0])
                ),
                "lines": lines,
                "text": block_text,
            })

    return _sort_blocks_by_layout(
        extracted,
        page.rect.width,
    )


def _extract_word_coordinates(page):
    """Extract individual native words and their coordinates."""

    words = page.get_text(
        "words",
        sort=False,
    )

    extracted_words = []

    for word in words:

        if len(word) < 8:
            continue

        (
            x0,
            y0,
            x1,
            y1,
            text,
            block_no,
            line_no,
            word_no,
        ) = word[:8]

        if not str(text).strip():
            continue

        block_no = int(block_no)

        extracted_words.append({
            "text": str(text),
            "bbox": [
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            ],
            "block_no": block_no,
            "block_id": (
                f"page-{page.number + 1}-block-{block_no}"
            ),
            "line_no": int(line_no),
            "word_no": int(word_no),
            "coordinate_system": "pymupdf_unrotated",
        })

    return extracted_words


def _extract_native_page(page):
    """Extract a digital page using PyMuPDF."""

    page_data = page.get_text(
        "dict",
        sort=False,
    )

    blocks = _extract_ordered_blocks(
        page,
        page_data,
    )

    words = _extract_word_coordinates(page)

    lines = [
        line
        for block in blocks
        for line in block["lines"]
    ]

    text = "\n\n".join(
        block["text"]
        for block in blocks
    ).strip()

    return {
        "text": text,
        "blocks": blocks,
        "lines": lines,
        "words": words,
        "line_count": len(lines),
        "word_count": len(words),
        "coordinate_system": "pymupdf_unrotated",
    }



def _map_ocr_bbox_to_pdf(
    bbox,
    page,
    image_width,
    image_height,
):
    """
    Convert OCR image coordinates into PyMuPDF's unrotated
    page coordinate system.

    First map pixels to the rendered page coordinate system.
    Then apply PyMuPDF's derotation matrix.
    """

    if not bbox or len(bbox) != 4:
        return None

    if image_width <= 0 or image_height <= 0:
        return None

    x0, y0, x1, y1 = bbox

    rotated_rect = pymupdf.Rect(
        x0 * page.rect.width / image_width,
        y0 * page.rect.height / image_height,
        x1 * page.rect.width / image_width,
        y1 * page.rect.height / image_height,
    )

    unrotated_rect = (
        rotated_rect * page.derotation_matrix
    )

    return [
        float(unrotated_rect.x0),
        float(unrotated_rect.y0),
        float(unrotated_rect.x1),
        float(unrotated_rect.y1),
    ]



def _extract_ocr_page(page):
    """
    Render a scanned PDF page and extract text using PaddleOCR.

    OCR provides line-level boxes. We preserve original image
    coordinates and additionally map them to PDF coordinates.

    We do not fabricate word-level coordinates.
    """

    matrix = pymupdf.Matrix(
        RENDER_SCALE,
        RENDER_SCALE,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image_width = pixmap.width
    image_height = pixmap.height

    with NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as temp_file:

        image_path = Path(temp_file.name)

    try:

        pixmap.save(str(image_path))

        ocr_result = extract_text_from_image(
            str(image_path)
        )

    finally:
        image_path.unlink(missing_ok=True)

    ocr_lines = []
    blocks = []

    for index, line in enumerate(
        ocr_result.get("lines", [])
    ):

        original_bbox = line.get("bbox")

        pdf_bbox = _map_ocr_bbox_to_pdf(
            bbox=original_bbox,
            page=page,
            image_width=image_width,
            image_height=image_height,
        )

        block_no = index

        block_id = (
            f"page-{page.number + 1}-block-{block_no}"
        )

        line_data = {
            "line_no": index,
            "block_no": block_no,
            "block_id": block_id,
            "text": line["text"],
            "image_bbox": _to_float_bbox(
                original_bbox
            ),
            "bbox": _to_float_bbox(pdf_bbox),
            "confidence": line.get("confidence"),
            "coordinate_system": "pymupdf_unrotated",
            "original_coordinate_system": "image_pixels",
            "spans": [],
        }

        ocr_lines.append(line_data)

        blocks.append({
            "block_id": block_id,
            "block_no": block_no,
            "bbox": line_data["bbox"],
            "image_bbox": line_data["image_bbox"],
            "lines": [
                {
                    "text": line_data["text"],
                    "bbox": line_data["bbox"],
                    "image_bbox": line_data["image_bbox"],
                    "confidence": line_data["confidence"],
                    "spans": [],
                }
            ],
            "text": line_data["text"],
            "confidence": line_data["confidence"],
            "coordinate_system": "pymupdf_unrotated",
            "original_coordinate_system": "image_pixels",
        })

    ordered_blocks = _sort_blocks_by_layout(
        blocks,
        page.rect.width,
    )

    ordered_lines = []

    for block in ordered_blocks:
        ordered_lines.extend(block["lines"])

    extracted_text = "\n\n".join(
        block["text"]
        for block in ordered_blocks
    ).strip()

    return {
        "text": extracted_text,
        "blocks": ordered_blocks,
        "lines": ordered_lines,
        "words": [],
        "line_count": len(ordered_lines),
        "word_count": 0,
        "coordinate_system": "pymupdf_unrotated",
        "original_coordinate_system": "image_pixels",
        "image_width": image_width,
        "image_height": image_height,
        "render_scale": RENDER_SCALE,
    }


def extract_pages(
    pdf_bytes: bytes,
    min_native_text_length: int = MIN_NATIVE_TEXT_LENGTH,
) -> list[dict]:
    """
    Extract all PDF pages using hybrid extraction.

    Digital pages:
        PyMuPDF native text, blocks, lines, and word coordinates.

    Scanned pages:
        PaddleOCR text, line coordinates, and confidence scores.

    Both extraction methods return a consistent page structure.
    """

    if not pdf_bytes:
        raise ValueError("PDF data is empty.")

    if min_native_text_length < 1:
        raise ValueError(
            "min_native_text_length must be at least 1."
        )

    pages = []

    with pymupdf.open(
        stream=io.BytesIO(pdf_bytes),
        filetype="pdf",
    ) as document:

        for index, page in enumerate(document):

            native_text = page.get_text(
                "text"
            ).strip()

            if len(native_text) >= min_native_text_length:

                result = _extract_native_page(page)
                extraction_source = "native"

            else:

                result = _extract_ocr_page(page)
                extraction_source = "ocr"

            pages.append({
                "page_number": index + 1,

                "text": result["text"],
                "blocks": result["blocks"],
                "lines": result["lines"],
                "words": result["words"],

                "line_count": result["line_count"],
                "word_count": result["word_count"],

                "extraction_source": extraction_source,

                "page_width": float(
                    page.cropbox.width
                ),
                "page_height": float(
                    page.cropbox.height
                ),
                "page_rotation": int(page.rotation),

                "coordinate_system": result[
                    "coordinate_system"
                ],

                "original_coordinate_system": result.get(
                    "original_coordinate_system"
                ),

                "image_width": result.get(
                    "image_width"
                ),

                "image_height": result.get(
                    "image_height"
                ),

                "render_scale": result.get(
                    "render_scale"
                ),
            })

    return pages


def extract_pdf_toc(pdf_bytes):
    """Extract embedded PDF bookmarks, if available."""

    with pymupdf.open(
        stream=io.BytesIO(pdf_bytes),
        filetype="pdf",
    ) as document:

        return document.get_toc()
