"""S3 access: unsigned/signed clients, byte-range GETs, FITS object listing."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config

FITS_SUFFIXES = (".fits", ".fit", ".fts")


@dataclass(frozen=True)
class S3Object:
    uri: str
    size: int
    etag: str


def parse_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into bucket and key."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri!r}")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket:
        raise ValueError(f"no bucket in uri: {uri!r}")
    return bucket, key


def make_client(anon: bool = False) -> Any:
    """boto3 S3 client with adaptive retries; unsigned when ``anon``."""
    kwargs: dict[str, Any] = {"retries": {"max_attempts": 5, "mode": "adaptive"}}
    if anon:
        kwargs["signature_version"] = UNSIGNED
    return boto3.client("s3", config=Config(**kwargs))


class S3Reader:
    """Thin wrapper: listing plus ranged reads. Safe to share across threads."""

    def __init__(self, client: Any | None = None, *, anon: bool = False) -> None:
        self.client = client if client is not None else make_client(anon)
        self.bytes_read = 0

    def list_fits(self, prefix_uri: str) -> Iterator[S3Object]:
        """Yield FITS objects under an ``s3://bucket/prefix`` location."""
        bucket, prefix = parse_uri(prefix_uri)
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", ()):
                key = obj["Key"]
                if not key.lower().endswith(FITS_SUFFIXES):
                    continue
                yield S3Object(
                    uri=f"s3://{bucket}/{key}",
                    size=int(obj["Size"]),
                    etag=str(obj.get("ETag", "")).strip('"'),
                )

    def ranged_get(self, uri: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes at ``offset``. Short reads at EOF are normal."""
        if length <= 0:
            return b""
        bucket, key = parse_uri(uri)
        end = offset + length - 1
        response = self.client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{end}")
        data: bytes = response["Body"].read()
        self.bytes_read += len(data)
        return data

    def fetcher(self, uri: str) -> UriFetch:
        """A :class:`~fitsq.fits_lite.Fetch` bound to one object."""
        return UriFetch(self, uri)


@dataclass(frozen=True)
class UriFetch:
    reader: S3Reader
    uri: str

    def __call__(self, offset: int, length: int) -> bytes:
        return self.reader.ranged_get(self.uri, offset, length)
