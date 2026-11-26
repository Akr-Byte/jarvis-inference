"""Read-only model discovery and metadata for registered and on-disk models."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from inference.config import settings
from inference.gguf import read_metadata
from inference.models import model_manager

SPLIT_SUFFIX = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})$")
QUANT = re.compile(r"(?i)(IQ[1-6]_[A-Z0-9]+|Q[2-8]_[A-Z0-9]+|F16|F32|BF16)")

# A short TTL avoids repeating a full rglob+stat on every status/metadata
# request while still noticing models added or removed during operation.
DISCOVERY_TTL_SECONDS = 2.0
_discovery_cache: dict[Path, tuple[float, list[DiscoveredModel]]] = {}


def invalidate_discovery_cache() -> None:
    _discovery_cache.clear()


@dataclass(frozen=True)
class DiscoveredModel:
    id: str
    path: str
    size_bytes: int
    shards: tuple[str, ...]
    metadata: dict[str, object]


def discover_gguf(directory: Path | None = None) -> list[DiscoveredModel]:
    root = (directory or settings.model_directory).expanduser()
    cached = _discovery_cache.get(root)
    now = time.monotonic()
    if cached is not None and now - cached[0] < DISCOVERY_TTL_SECONDS:
        return cached[1]
    if not root.is_dir():
        _discovery_cache[root] = (now, [])
        return []
    files = sorted(path for path in root.rglob("*.gguf") if path.is_file())
    grouped: dict[str, list[tuple[Path, int, int]]] = {}
    ordinary: list[Path] = []
    for path in files:
        match = SPLIT_SUFFIX.match(path.stem)
        if match:
            grouped.setdefault(match.group("base"), []).append(
                (path, int(match.group("index")), int(match.group("count")))
            )
        else:
            ordinary.append(path)
    entries: list[DiscoveredModel] = []
    for path in ordinary:
        entries.append(_entry(path, (path,)))
    for base, shards in grouped.items():
        shards.sort(key=lambda item: item[1])
        expected = shards[0][2]
        if len(shards) != expected or [item[1] for item in shards] != list(range(1, expected + 1)):
            continue
        paths = tuple(item[0] for item in shards)
        entries.append(_entry(paths[0], paths, model_id=base))
    discovered = sorted(entries, key=lambda model: model.id.casefold())
    _discovery_cache[root] = (now, discovered)
    return discovered


def _entry(first: Path, shards: tuple[Path, ...], model_id: str | None = None) -> DiscoveredModel:
    try:
        metadata = read_metadata(first)
    except (OSError, ValueError) as exc:
        metadata = {"metadata_error": str(exc)}
    try:
        size_bytes = sum(path.stat().st_size for path in shards)
    except OSError:
        size_bytes = 0
    return DiscoveredModel(
        id=model_id or first.stem,
        path=str(first),
        size_bytes=size_bytes,
        shards=tuple(str(path) for path in shards),
        metadata=metadata,
    )


def model_details(model_id: str) -> dict[str, object] | None:
    from inference.llamacpp_runtime import runtime

    registered = model_id in settings.inference_models
    discovered = {model.id: model for model in discover_gguf()}
    disk_model = discovered.get(model_id)
    configured_path = settings.llamacpp_model_map.get(model_id)
    if configured_path:
        path = Path(configured_path).expanduser()
        disk_model = next(
            (entry for entry in discover_gguf(path.parent) if Path(entry.path) == path),
            None,
        ) or _entry(path, (path,), model_id=model_id)
    is_runtime_model = bool(
        settings.llamacpp_model
        and (
            Path(settings.llamacpp_model).expanduser().stem == model_id
            or settings.llamacpp_alias == model_id
        )
    )
    if disk_model is None and is_runtime_model:
        path = Path(settings.llamacpp_model).expanduser()
        disk_model = next(
            (entry for entry in discover_gguf(path.parent) if Path(entry.path) == path),
            None,
        ) or _entry(path, (path,), model_id=model_id)
    if not registered and disk_model is None:
        return None
    model_state = model_manager.status()
    loaded = model_id in model_state["loaded_models"]
    available = model_id in model_state["available_models"]
    metadata = disk_model.metadata if disk_model else {}
    quant_match = QUANT.search(disk_model.path if disk_model else model_id)
    architecture = str(metadata.get("general.architecture") or "llama")
    context_capability = metadata.get(f"{architecture}.context_length")
    runtime_state = runtime.status()
    return {
        "id": model_id,
        "display_name": str(metadata.get("general.name") or model_id),
        "path": disk_model.path if disk_model else None,
        "architecture": metadata.get("general.architecture"),
        "quantization": quant_match.group(0) if quant_match else None,
        "file_size_bytes": disk_model.size_bytes if disk_model else None,
        "split_shards": list(disk_model.shards) if disk_model else [],
        "context_capability": context_capability,
        "effective_context_limit": settings.max_context_length,
        "recommended_context": min(int(context_capability), settings.max_context_length)
        if context_capability
        else settings.max_context_length,
        "backend": "llama.cpp" if settings.llamacpp_base_url else "ollama",
        "runtime_profile": "workstation" if settings.llamacpp_managed else "default",
        "runtime_cache": {
            "prompt_cache": settings.llamacpp_cache_prompt if settings.llamacpp_managed else None,
            "cache_ram_mib": settings.llamacpp_cache_ram_mib if settings.llamacpp_managed else None,
            "kv_cache_type_k": settings.llamacpp_cache_type_k or "f16"
            if settings.llamacpp_managed
            else None,
            "kv_cache_type_v": settings.llamacpp_cache_type_v or "f16"
            if settings.llamacpp_managed
            else None,
        },
        "estimated_memory_bytes": int(float(runtime_state["estimated_memory_gib"]) * 1024**3)
        if runtime_state["model_id"] == model_id
        and runtime_state["estimated_memory_gib"] is not None
        else None,
        "default_max_output_tokens": settings.max_output_tokens,
        "enabled": registered,
        "discovered_on_disk": disk_model is not None,
        "registered": registered,
        "loaded": loaded,
        "available": available,
        "accepting_requests": registered and (loaded if settings.llamacpp_managed else available),
        "metadata": metadata,
    }


def registry_status() -> dict[str, object]:
    discovered = discover_gguf()
    registered = set(settings.inference_models)
    state = model_manager.status()
    loaded = set(state["loaded_models"])
    available = set(state["available_models"])
    return {
        "discovered_models": [asdict(item) for item in discovered],
        "registered_models": sorted(registered),
        "loaded_models": sorted(loaded),
        "available_models": sorted(registered & available),
        "accepting_models": sorted(registered & loaded)
        if settings.llamacpp_managed
        else sorted(registered & available),
    }
