"""Small, dependency-free reader for selected GGUF header metadata."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

GGUF_TYPES = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}
WANTED = {
    "general.architecture",
    "general.name",
    "general.file_type",
    "llama.context_length",
    "llama.block_count",
    "llama.embedding_length",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.attention.key_length",
    "llama.attention.value_length",
}


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return data


def _u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _string(stream: BinaryIO) -> str:
    length = _u64(stream)
    if length > 1_000_000:
        raise ValueError("invalid GGUF string length")
    return _read_exact(stream, length).decode("utf-8", errors="replace")


def _skip_value(stream: BinaryIO, kind: int) -> object | None:
    if kind == 8:
        return _string(stream)
    if kind == 9:
        element_type = _u32(stream)
        count = _u64(stream)
        if count > 10_000_000:
            raise ValueError("invalid GGUF array length")
        if element_type == 8:
            for _ in range(count):
                _string(stream)
        elif element_type == 9:
            raise ValueError("nested GGUF arrays are invalid")
        elif element_type in GGUF_TYPES:
            stream.seek(count * GGUF_TYPES[element_type][1], 1)
        else:
            raise ValueError(f"unsupported GGUF value type: {element_type}")
        return None
    if kind not in GGUF_TYPES:
        raise ValueError(f"unsupported GGUF value type: {kind}")
    code, size = GGUF_TYPES[kind]
    return struct.unpack("<" + code, _read_exact(stream, size))[0]


def read_metadata(path: Path) -> dict[str, object]:
    """Read scalar model metadata without loading tensor payloads."""
    with path.open("rb") as stream:
        if _read_exact(stream, 4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = _u32(stream)
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version: {version}")
        _u64(stream)  # tensor count
        metadata_count = _u64(stream)
        if metadata_count > 1_000_000:
            raise ValueError("invalid GGUF metadata count")
        result: dict[str, object] = {"gguf_version": version}
        for _ in range(metadata_count):
            key = _string(stream)
            kind = _u32(stream)
            value = _skip_value(stream, kind)
            if (
                key in WANTED
                or key.endswith(
                    (
                        ".context_length",
                        ".block_count",
                        ".embedding_length",
                        ".attention.head_count",
                        ".attention.head_count_kv",
                        ".attention.key_length",
                        ".attention.value_length",
                    )
                )
            ) and value is not None:
                result[key] = value
        return result


def kv_bytes_per_token(
    metadata: dict[str, object], cache_type: str = "f16", cache_type_v: str | None = None
) -> int | None:
    architecture = str(metadata.get("general.architecture") or "llama")

    def get(key: str, default: object = None) -> object:
        return metadata.get(f"{architecture}.{key}", default)

    try:
        layers = int(get("block_count"))
        heads = int(get("attention.head_count_kv"))
        key_dim = int(get("attention.key_length", 0))
        value_dim = int(get("attention.value_length", 0))
        if not key_dim or not value_dim:
            embedding = int(get("embedding_length"))
            query_heads = int(get("attention.head_count"))
            key_dim = value_dim = embedding // query_heads
        if not all(value > 0 for value in (layers, heads, key_dim, value_dim)):
            return None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    bytes_per_element = {
        "f16": 2.0,
        "f32": 4.0,
        "bf16": 2.0,
        "q8_0": 1.0625,
        "q5_0": 0.6875,
        "q5_1": 0.75,
        "q4_0": 0.5625,
        "q4_1": 0.625,
    }
    bytes_k = bytes_per_element.get(cache_type.lower())
    bytes_v = bytes_per_element.get((cache_type_v or cache_type).lower())
    if bytes_k is None or bytes_v is None:
        return None
    return int(layers * heads * (key_dim * bytes_k + value_dim * bytes_v))
