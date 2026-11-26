"""Small, dependency-free reader for selected GGUF header metadata."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
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


def _skip_string(stream: BinaryIO) -> None:
    """Advance past a string without decoding or allocating it.

    Tokenizer vocabularies hold ~10^5 length-prefixed strings; materialising
    them is pure allocation churn for a header scan that discards the values.
    """
    length = _u64(stream)
    if length > 1_000_000:
        raise ValueError("invalid GGUF string length")
    stream.seek(length, 1)


def _skip_value(stream: BinaryIO, kind: int, *, keep_array: bool = False) -> object | None:
    if kind == 8:
        return _string(stream)
    if kind == 9:
        element_type = _u32(stream)
        count = _u64(stream)
        if count > 10_000_000:
            raise ValueError("invalid GGUF array length")
        if element_type == 8:
            if keep_array:
                return [_string(stream) for _ in range(count)]
            for _ in range(count):
                _skip_string(stream)
            return None
        if element_type == 9:
            raise ValueError("nested GGUF arrays are invalid")
        if element_type in GGUF_TYPES:
            if keep_array:
                code, size = GGUF_TYPES[element_type]
                return [
                    struct.unpack("<" + code, _read_exact(stream, size))[0] for _ in range(count)
                ]
            stream.seek(count * GGUF_TYPES[element_type][1], 1)
            return None
        raise ValueError(f"unsupported GGUF value type: {element_type}")
    if kind not in GGUF_TYPES:
        raise ValueError(f"unsupported GGUF value type: {kind}")
    code, size = GGUF_TYPES[kind]
    return struct.unpack("<" + code, _read_exact(stream, size))[0]


@lru_cache(maxsize=256)
def _read_metadata_cached(path: str, size: int, mtime_ns: int) -> dict[str, object]:
    return _read_metadata_file(Path(path))


def read_metadata(path: Path) -> dict[str, object]:
    """Read scalar model metadata without loading tensor payloads.

    Results are cached by path and (size, mtime) so registry/status endpoints
    do not re-parse every GGUF header on each request.
    """
    stat = path.stat()
    return dict(_read_metadata_cached(str(path), stat.st_size, stat.st_mtime_ns))


def _read_metadata_file(path: Path) -> dict[str, object]:
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
                stream,
                kind,
                keep_array=key.endswith(
                    (".attention.recurrent_layers", ".attention.head_count_kv")
                ),
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


# Architecture names that are known to combine attention with recurrent
# (SSM/Mamba) layers. They must provide explicit layer metadata; guessing a
# conventional all-attention layout would understate memory.
HYBRID_ARCHITECTURE_MARKERS = ("mamba", "hybrid", "rwkv", "jamba", "falcon_h1")
# Architectures in this set are known to combine attention with recurrent/SSM layers,
# and require SSM metadata (conv_kernel, inner_size, state_size, etc.) to be present.
HYBRID_ARCHITECTURES_REQUIRING_SSM = {"qwen3next"}


def _kv_layout(
    metadata: dict[str, object], architecture: str, layers: int
) -> tuple[list[int], int] | None:
    """Return per-layer KV head counts plus the recurrent-layer count.

    Handles three metadata shapes used by llama.cpp:
    * scalar ``attention.head_count_kv`` with a ``recurrent_layers`` mask,
    * per-layer ``attention.head_count_kv`` where recurrent layers are zero,
    * scalar heads with a ``full_attention_interval`` fallback.

    Returns ``None`` when a hybrid architecture cannot be mapped, so admission
    fails closed instead of assuming an all-attention KV cache.
    """
    raw_heads = metadata.get(f"{architecture}.attention.head_count_kv")
    if isinstance(raw_heads, list):
        if len(raw_heads) != layers:
            return None
        if not all(isinstance(value, int) and value >= 0 for value in raw_heads):
            return None
        heads = [int(value) for value in raw_heads]
    else:
        try:
            scalar = int(raw_heads)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if scalar <= 0:
            return None
        heads = [scalar] * layers
    layer_map = metadata.get(f"{architecture}.attention.recurrent_layers")
    interval = metadata.get(f"{architecture}.full_attention_interval")
    has_ssm = any(key.startswith(f"{architecture}.ssm.") for key in metadata)
    hybrid_hint = (
        has_ssm
        or isinstance(layer_map, (list, tuple))
        or any(marker in architecture.lower() for marker in HYBRID_ARCHITECTURE_MARKERS)
    )
    # For architectures known to combine attention with recurrent/SSM layers,
    # SSM metadata must be present; otherwise we cannot reliably estimate KV.
    if architecture in HYBRID_ARCHITECTURES_REQUIRING_SSM and not has_ssm:
        return None
    if isinstance(layer_map, list):
        if len(layer_map) != layers:
            return None
        heads = [0 if recurrent else heads[index] for index, recurrent in enumerate(layer_map)]
    elif any(value == 0 for value in heads):
        pass  # per-layer zero heads already identify recurrent layers
    elif hybrid_hint:
        if not isinstance(interval, int) or interval <= 0:
            return None
        # llama.cpp's fallback marks every interval-th layer (from the end of
        # the first interval) as full attention.
        heads = [heads[index] if (index + 1) % interval == 0 else 0 for index in range(layers)]
    recurrent = sum(1 for value in heads if value == 0)
    attention = layers - recurrent
    if recurrent and not has_ssm:
        return None
    if hybrid_hint and not recurrent:
        return None
    if sum(1 for value in heads if value > 0) != attention:
        return None
    return heads, recurrent


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
    architecture = str(metadata.get("general.architecture") or "llama")
    try:
        layers = int(metadata.get(f"{architecture}.block_count", 0))
    except (TypeError, ValueError):
        return KvEstimate(None, None, None, None, False, "invalid block count metadata")
    if layers <= 0:
        return KvEstimate(
            None, None, None, None, False, "required architecture metadata unavailable"
        )
    layout = _kv_layout(metadata, architecture, layers)
    if layout is None:
        return KvEstimate(
            None, None, None, None, False, "required architecture metadata unavailable"
        )
    heads, recurrent_layers = layout
    attention_layers = layers - recurrent_layers
    per_token = _kv_bytes_per_token_for_layout(
        metadata, architecture, heads, cache_type, cache_type_v
    )
    if per_token is None:
        return KvEstimate(
            None, None, None, None, False, "required architecture metadata unavailable"
        )
    recurrent_per_sequence = 0
    if recurrent_layers:
        recurrent_per_sequence = _hybrid_recurrent_bytes(metadata, architecture, recurrent_layers)
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


def _hybrid_recurrent_bytes(
    metadata: dict[str, object], architecture: str, recurrent_count: int
) -> int | None:
    """Per-sequence recurrent state (conv + SSM) for gated-delta/Mamba layers.

    ``state_size * inner_size`` equals the Mamba2 ``nheads * head_dim * d_state``
    state because ``inner_size`` is the SSM inner dimension. The convolution
    term is deliberately doubled, which over- rather than under-estimates.
    """
    try:
        conv_kernel = int(metadata[f"{architecture}.ssm.conv_kernel"])
        inner_size = int(metadata[f"{architecture}.ssm.inner_size"])
        state_size = int(metadata[f"{architecture}.ssm.state_size"])
        group_count = int(metadata[f"{architecture}.ssm.group_count"])
        time_step_rank = int(metadata.get(f"{architecture}.ssm.time_step_rank", 1))
        if not all(v > 0 for v in (conv_kernel, inner_size, state_size, group_count)):
            return None
        if time_step_rank > 0 and group_count > 0 and time_step_rank % group_count:
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
    try:
        layers = int(metadata.get(f"{architecture}.block_count", 0))
    except (TypeError, ValueError):
        return None
    if layers <= 0:
        return None
    layout = _kv_layout(metadata, architecture, layers)
    if layout is None:
        return None
    heads, _ = layout
    return _kv_bytes_per_token_for_layout(metadata, architecture, heads, cache_type, cache_type_v)


def _kv_bytes_per_token_for_layout(
    metadata: dict[str, object],
    architecture: str,
    heads: list[int],
    cache_type: str,
    cache_type_v: str | None,
) -> int | None:
    def get(key: str, default: object = None) -> object:
        return metadata.get(f"{architecture}.{key}", default)

    try:
        key_dim = int(get("attention.key_length", 0))
        value_dim = int(get("attention.value_length", 0))
        if not key_dim or not value_dim:
            embedding = int(get("embedding_length"))
            query_heads = int(get("attention.head_count"))
            key_dim = value_dim = embedding // query_heads
        if not all(value > 0 for value in (key_dim, value_dim)):
            return None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    bytes_k = _cache_bytes_per_element(cache_type)
    bytes_v = _cache_bytes_per_element(cache_type_v or cache_type)
    if bytes_k is None or bytes_v is None:
        return None
    total_heads = sum(heads)
    if total_heads <= 0:
        return None
    return int(total_heads * (key_dim * bytes_k + value_dim * bytes_v))
