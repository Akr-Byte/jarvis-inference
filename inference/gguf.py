"""Small, dependency-free reader for selected GGUF header metadata."""

from __future__ import annotations

import struct
from dataclasses import dataclass
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
    "qwen3next.full_attention_interval",
    "qwen3next.attention.recurrent_layers",
    "qwen3next.ssm.conv_kernel",
    "qwen3next.ssm.inner_size",
    "qwen3next.ssm.state_size",
    "qwen3next.ssm.time_step_rank",
    "qwen3next.ssm.group_count",
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


def _skip_value(stream: BinaryIO, kind: int, *, keep_array: bool = False) -> object | None:
    if kind == 8:
        return _string(stream)
    if kind == 9:
        element_type = _u32(stream)
        count = _u64(stream)
        if count > 10_000_000:
            raise ValueError("invalid GGUF array length")
        if element_type == 8:
            values = [] if keep_array else None
            for _ in range(count):
                value = _string(stream)
                if values is not None:
                    values.append(value)
        elif element_type == 9:
            raise ValueError("nested GGUF arrays are invalid")
        elif element_type in GGUF_TYPES:
            if keep_array:
                code, size = GGUF_TYPES[element_type]
                values = [
                    struct.unpack("<" + code, _read_exact(stream, size))[0] for _ in range(count)
                ]
            else:
                stream.seek(count * GGUF_TYPES[element_type][1], 1)
                values = None
        else:
            raise ValueError(f"unsupported GGUF value type: {element_type}")
        return values if keep_array else None
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
            value = _skip_value(
                stream, kind, keep_array=key.endswith(".attention.recurrent_layers")
            )
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
                        ".full_attention_interval",
                        ".attention.recurrent_layers",
                        ".ssm.conv_kernel",
                        ".ssm.inner_size",
                        ".ssm.state_size",
                        ".ssm.time_step_rank",
                        ".ssm.group_count",
                    )
                )
            ) and value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class KvEstimate:
    bytes_total: int | None
    attention_layers: int | None
    recurrent_layers: int | None
    recurrent_state_bytes_per_sequence: int | None
    complete: bool
    reason: str | None = None


def estimate_kv_memory(
    metadata: dict[str, object],
    context_tokens: int,
    parallel_sequences: int = 1,
    cache_type: str = "f16",
    cache_type_v: str | None = None,
) -> KvEstimate:
    """Estimate persistent KV/recurrent memory; unknown hybrid shapes fail closed."""
    if context_tokens <= 0 or parallel_sequences <= 0:
        return KvEstimate(None, None, None, None, False, "context and parallelism must be positive")
    per_token = kv_bytes_per_token(metadata, cache_type, cache_type_v)
    if per_token is None:
        return KvEstimate(
            None, None, None, None, False, "required architecture metadata unavailable"
        )
    architecture = str(metadata.get("general.architecture") or "llama")
    layers = int(metadata.get(f"{architecture}.block_count", 0))
    recurrent_layers = None
    recurrent_per_sequence = 0
    attention_layers = layers
    if architecture == "qwen3next":
        layer_map = metadata.get(f"{architecture}.attention.recurrent_layers")
        if isinstance(layer_map, list):
            recurrent_layers = sum(bool(value) for value in layer_map)
            attention_layers = layers - recurrent_layers
            recurrent_per_sequence = _qwen3next_recurrent_bytes(metadata, recurrent_layers)
            if recurrent_per_sequence is None:
                return KvEstimate(
                    None,
                    attention_layers,
                    recurrent_layers,
                    None,
                    False,
                    "invalid recurrent state metadata",
                )
        else:
            interval = metadata.get(f"{architecture}.full_attention_interval")
            if not isinstance(interval, int) or interval <= 0:
                return KvEstimate(
                    None, None, None, None, False, "hybrid layer layout metadata unavailable"
                )
            attention_layers = layers // interval
            recurrent_layers = layers - attention_layers
            recurrent_per_sequence = _qwen3next_recurrent_bytes(metadata, recurrent_layers)
            if recurrent_per_sequence is None:
                return KvEstimate(
                    None,
                    attention_layers,
                    recurrent_layers,
                    None,
                    False,
                    "invalid recurrent state metadata",
                )
    total = (
        per_token * context_tokens * parallel_sequences
        + recurrent_per_sequence * parallel_sequences
    )
    return KvEstimate(total, attention_layers, recurrent_layers, recurrent_per_sequence, True)


def _qwen3next_recurrent_bytes(metadata: dict[str, object], recurrent_count: int) -> int | None:
    architecture = "qwen3next"
    try:
        conv_kernel = int(metadata[f"{architecture}.ssm.conv_kernel"])
        inner_size = int(metadata[f"{architecture}.ssm.inner_size"])
        state_size = int(metadata[f"{architecture}.ssm.state_size"])
        group_count = int(metadata[f"{architecture}.ssm.group_count"])
        if not all(v > 0 for v in (conv_kernel, inner_size, state_size, group_count)):
            return None
        return (
            recurrent_count
            * 4
            * (
                conv_kernel * (2 * state_size * group_count + 2 * inner_size)
                + state_size * inner_size
            )
        )
    except (KeyError, TypeError, ValueError):
        return None


def _cache_bytes_per_element(value: str) -> float:
    return {
        "f16": 2.0,
        "f32": 4.0,
        "bf16": 2.0,
        "q8_0": 1.0625,
        "q5_0": 0.6875,
        "q5_1": 0.75,
        "q4_0": 0.5625,
        "q4_1": 0.625,
    }.get(value.lower())


def kv_bytes_per_token(
    metadata: dict[str, object], cache_type: str = "f16", cache_type_v: str | None = None
) -> int | None:
    architecture = str(metadata.get("general.architecture") or "llama")
    architecture_lower = architecture.lower()
    if architecture != "qwen3next" and (
        any(
            marker in architecture_lower
            for marker in ("mamba", "hybrid", "rwkv", "jamba", "falcon_h1")
        )
        or f"{architecture}.attention.recurrent_layers" in metadata
        or any(key.startswith(f"{architecture}.ssm.") for key in metadata)
    ):
        return None

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
    bytes_k = _cache_bytes_per_element(cache_type)
    bytes_v = _cache_bytes_per_element(cache_type_v or cache_type)
    if bytes_k is None or bytes_v is None:
        return None
    full_attention_layers = layers
    if architecture == "qwen3next":
        recurrent_layers = get("attention.recurrent_layers")
        if isinstance(recurrent_layers, list):
            if len(recurrent_layers) != layers:
                return None
            recurrent_count = sum(bool(value) for value in recurrent_layers)
            full_attention_layers = layers - recurrent_count
            try:
                time_step_rank = int(get("ssm.time_step_rank"))
                group_count = int(get("ssm.group_count"))
                if time_step_rank <= 0 or group_count <= 0 or time_step_rank % group_count:
                    return None
                if _qwen3next_recurrent_bytes(metadata, recurrent_count) is None:
                    return None
            except (TypeError, ValueError):
                return None
        else:
            try:
                interval = int(get("full_attention_interval"))
                time_step_rank = int(get("ssm.time_step_rank"))
                group_count = int(get("ssm.group_count"))
                if interval <= 0:
                    return None
                if time_step_rank <= 0 or group_count <= 0 or time_step_rank % group_count:
                    return None
                # llama.cpp's Qwen3-Next fallback marks each interval-th layer
                # full attention, starting at the interval boundary.
                full_attention_layers = layers // interval
                recurrent_count = layers - full_attention_layers
            except (TypeError, ValueError):
                return None
            if _qwen3next_recurrent_bytes(metadata, recurrent_count) is None:
                return None
    return int(full_attention_layers * heads * (key_dim * bytes_k + value_dim * bytes_v))
