from __future__ import annotations

import io
import os
import re
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

import httpx

_URL_SCHEMES = {"http", "https"}
_CONTENT_DISPOSITION_RE = re.compile(
    r"filename\*\s*=\s*(?:UTF-8'')?(?P<extended>[^;]+)"
    r"|filename\s*=\s*\"?(?P<plain>[^\";]+)\"?",
    re.IGNORECASE,
)


def is_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme.lower() in _URL_SCHEMES and bool(parsed.netloc)


def _filename_from_content_disposition(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    match = _CONTENT_DISPOSITION_RE.search(header)
    if not match:
        return None
    raw = match.group("extended") or match.group("plain")
    if raw is None:
        return None
    return unquote(raw.strip())


def filename_from_url(url: str, content_disposition: Optional[str] = None) -> str:
    name = _filename_from_content_disposition(content_disposition)
    if not name:
        path = urlparse(url).path
        name = unquote(os.path.basename(path))
    if not name:
        name = "downloaded"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name


def download_pdf_to_memory(
    url: str,
    *,
    timeout: float = 60.0,
    chunk_size: int = 65536,
    max_bytes: Optional[int] = None,
) -> Tuple[io.BytesIO, str]:
    """Stream a PDF from `url` into an in-memory BytesIO.

    Returns (buffer rewound to 0, filename derived from URL/Content-Disposition).
    Raises httpx.HTTPError on transport failure and ValueError on size overflow.
    """
    buffer = io.BytesIO()
    total = 0
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        content_disposition = response.headers.get("content-disposition")
        for chunk in response.iter_bytes(chunk_size=chunk_size):
            if not chunk:
                continue
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError(
                    f"Download exceeded max_bytes={max_bytes} while fetching {url}"
                )
            buffer.write(chunk)
    buffer.seek(0)
    return buffer, filename_from_url(url, content_disposition)
