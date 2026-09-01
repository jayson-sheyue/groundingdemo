"""Upload multimodal files to GCS for Vertex AI (browser → bucket)."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from google.cloud import storage

from demo_grounded_search import _env, guess_mime_type

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".mp4",
    ".mov",
    ".mp3",
    ".wav",
    ".txt",
    ".csv",
}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # PDF limit per Google multimodal docs


def gcs_upload_prefix() -> str:
    return (_env("GCS_PREFIX") or "uploads").strip("/")


def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^\w.\-]+", "_", base)
    return base[:200] or "file"


def upload_bytes_to_gcs(
    data: bytes,
    filename: str,
    *,
    content_type: Optional[str] = None,
) -> dict[str, Any]:
    bucket_name = _env("GCS_BUCKET")
    if not bucket_name:
        raise RuntimeError("GCS_BUCKET is not set in .env")

    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")

    safe = _safe_filename(filename)
    ext = Path(safe).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type {ext!r}. Allowed: {allowed}")

    prefix = gcs_upload_prefix()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    object_name = f"{prefix}/{stamp}/{uuid.uuid4().hex[:8]}_{safe}"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    mime = content_type or guess_mime_type(safe)
    blob.upload_from_string(data, content_type=mime)

    return {
        "filename": safe,
        "object_name": object_name,
        "gcs_uri": f"gs://{bucket_name}/{object_name}",
        "mime_type": mime,
        "size_bytes": len(data),
    }
