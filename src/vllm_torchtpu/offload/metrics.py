# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Observability metrics and collectors for Raiden KV cache offloading.

In vLLM V1, the scheduler (`EngineCore`) and Prometheus `/metrics` endpoint
(`API Server`) run in separate processes, and Raiden executes all HBM <->
Host/Peer/Storage transfers from the scheduler-side `KVCacheStore` controller
(worker-side `get_finished()` is always empty):

- `RaidenOffloadStatsCollector` runs in `EngineCore`, tracking per-sub-block
  tier residency, deduplicated token counts, and per-tier transfer latencies,
  draining per-step deltas as `OffloadingConnectorStats` via `pop_stats()`.
- `RaidenOffloadPromMetrics` runs in the API server process, registering
  Raiden's `OffloadingSpec` metric definitions and applying IPC stats deltas to
  Prometheus counters and histograms.
"""

from __future__ import annotations

import copy
import enum
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    PromMetric,
    PromMetricT,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_registered_specs: set[str] = set()


# Load/hit metrics label where data is read from (`source`), while save metrics
# label where data is written to (`destination`), keeping `source` aligned with
# upstream `vllm:prompt_tokens_cached_by_source` for PromQL rollups.
OFFLOAD_LOAD_LABELNAMES: tuple[str, str] = ("source", "backend")
OFFLOAD_SAVE_LABELNAMES: tuple[str, str] = ("destination", "backend")
UNKNOWN_BACKEND_LABEL = "unknown"
HOST_OFFLOAD_LABELS: tuple[str, str] = (CacheHitSource.HOST.value, "dram")
P2P_OFFLOAD_LABELS: tuple[str, str] = (CacheHitSource.P2P.value, "rdma")
# Reserved strictly for unrecognized C++ `BlockStatus` values (never used for
# keys missing from `_step_key_sources`).
UNSPECIFIED_OFFLOAD_LABELS: tuple[str, str] = (
    CacheHitSource.EXTERNAL_UNSPECIFIED.value,
    UNKNOWN_BACKEND_LABEL,
)

# Covers sub-millisecond local DRAM DMAs (~250 us) up to multi-second peer RDMA
# or shared-storage reads under load.
OFFLOAD_LATENCY_BUCKETS: tuple[float, ...] = (
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


class RaidenOffloadMetricName(enum.StrEnum):
    """Directional Prometheus metrics emitted by Raiden KV offload."""

    HIT_TOKENS = "vllm:kv_offload_hit_tokens"
    LOADED_TOKENS = "vllm:kv_offload_loaded_tokens"
    SAVED_TOKENS = "vllm:kv_offload_saved_tokens"
    LOAD_LATENCY = "vllm:kv_offload_load_latency_seconds"
    SAVE_LATENCY = "vllm:kv_offload_save_latency_seconds"
    FAILED_LOAD_JOBS = "vllm:kv_offload_failed_load_jobs"
    FAILED_SAVE_JOBS = "vllm:kv_offload_failed_save_jobs"


class OffloadTransferMetricName(enum.StrEnum):
    """Upstream flat transfer metric names populated scheduler-side.

    Upstream `OffloadingConnector` normally populates these from worker-side
    `TransferResult`s, which Raiden never emits because transfers are driven by
    the scheduler-side controller. Recording these keys in
    `OffloadingConnectorStats` lets `OffloadPromMetrics` observe them directly.
    """

    LOAD_BYTES = "vllm:kv_offload_load_bytes"
    LOAD_TIME = "vllm:kv_offload_load_time"
    LOAD_SIZE = "vllm:kv_offload_load_size"
    STORE_BYTES = "vllm:kv_offload_store_bytes"
    STORE_TIME = "vllm:kv_offload_store_time"
    STORE_SIZE = "vllm:kv_offload_store_size"


def build_raiden_offload_metric_definitions(
    extra_config: Mapping[str, Any],
) -> dict[str, OffloadingMetricMetadata]:
    """Return Prometheus metric definitions for TPURaidenStoreOffloadingSpec."""
    del extra_config
    return {
        RaidenOffloadMetricName.HIT_TOKENS: OffloadingCounterMetadata(
            documentation=(
                "Number of prompt tokens matched and pinned in external KV "
                "offload tiers at scheduler admission, labeled by source and "
                "backend."
            ),
            labelnames=OFFLOAD_LOAD_LABELNAMES,
        ),
        RaidenOffloadMetricName.LOADED_TOKENS: OffloadingCounterMetadata(
            documentation=(
                "Number of KV cache tokens successfully loaded into device HBM "
                "from external offload tiers, labeled by source and backend."
            ),
            labelnames=OFFLOAD_LOAD_LABELNAMES,
        ),
        RaidenOffloadMetricName.SAVED_TOKENS: OffloadingCounterMetadata(
            documentation=(
                "Number of KV cache tokens successfully saved from device HBM "
                "to external offload tiers, labeled by destination and backend."
            ),
            labelnames=OFFLOAD_SAVE_LABELNAMES,
        ),
        RaidenOffloadMetricName.LOAD_LATENCY: OffloadingHistogramMetadata(
            documentation=(
                "End-to-end KV offload load job latency from transfer launch "
                "to polled completion in seconds, labeled by source and backend."
            ),
            labelnames=OFFLOAD_LOAD_LABELNAMES,
            buckets=OFFLOAD_LATENCY_BUCKETS,
        ),
        RaidenOffloadMetricName.SAVE_LATENCY: OffloadingHistogramMetadata(
            documentation=(
                "End-to-end KV offload save job latency from transfer launch "
                "to polled completion in seconds, labeled by destination and "
                "backend."
            ),
            labelnames=OFFLOAD_SAVE_LABELNAMES,
            buckets=OFFLOAD_LATENCY_BUCKETS,
        ),
        RaidenOffloadMetricName.FAILED_LOAD_JOBS: OffloadingCounterMetadata(
            documentation=(
                "Number of launched KV offload load jobs that failed, labeled "
                "by source and backend."
            ),
            labelnames=OFFLOAD_LOAD_LABELNAMES,
        ),
        RaidenOffloadMetricName.FAILED_SAVE_JOBS: OffloadingCounterMetadata(
            documentation=(
                "Number of KV offload save jobs that failed, labeled by "
                "destination and backend."
            ),
            labelnames=OFFLOAD_SAVE_LABELNAMES,
        ),
    }


def _ensure_spec_registered(spec_cls: type[OffloadingSpec]) -> str:
    """Idempotently register `spec_cls` in upstream `OffloadingSpecFactory`."""
    spec_name = spec_cls.__name__
    if spec_name in _registered_specs:
        return spec_name
    OffloadingSpecFactory.register_spec(
        spec_name,
        spec_cls.__module__,
        spec_name,
    )
    _registered_specs.add(spec_name)
    return spec_name


class RaidenOffloadPromMetrics(OffloadPromMetrics):
    """Prometheus metrics binder for TPURaidenOffloadingConnector.

    `TPURaidenOffloadingConnector` instantiates `TPURaidenStoreOffloadingSpec`
    directly (bypassing `OffloadingSpecFactory`), whereas upstream
    `OffloadPromMetrics.__init__` in the API server process resolves
    `spec_cls.build_metric_definitions` via `OffloadingSpecFactory` using
    `kv_connector_extra_config["spec_name"]`. This subclass registers `spec_cls`
    and injects `spec_name` on shallow config copies so the caller's
    `VllmConfig` and `KVTransferConfig` are never mutated.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        *,
        spec_cls: type[OffloadingSpec],
    ):
        spec_name = _ensure_spec_registered(spec_cls)
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        scoped_transfer_config = copy.copy(kv_transfer_config)
        scoped_transfer_config.kv_connector_extra_config = {
            **kv_transfer_config.kv_connector_extra_config,
            "spec_name": spec_name,
        }
        scoped_vllm_config = copy.copy(vllm_config)
        scoped_vllm_config.kv_transfer_config = scoped_transfer_config
        super().__init__(
            scoped_vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )


def status_to_hit_source(status: Any, block_status_cls: Any) -> CacheHitSource:
    """Map a Raiden C++ `BlockStatus` value to upstream `CacheHitSource`."""
    if status in (block_status_cls.HOST, block_status_cls.HOST_AND_HBM):
        return CacheHitSource.HOST
    if status == block_status_cls.REMOTE:
        return CacheHitSource.P2P
    if status == block_status_cls.SHARED_STORAGE:
        return CacheHitSource.DISK
    return CacheHitSource.EXTERNAL_UNSPECIFIED


def filter_positive_source_tokens(
    tokens_by_source: Mapping[CacheHitSource, int],
) -> dict[CacheHitSource, int]:
    """Filter per-source token counts to positive external `CacheHitSource`s.

    Checks `isinstance(source, CacheHitSource)` explicitly because
    `CacheHitSource` inherits from `str` (`"host" == CacheHitSource.HOST`), and
    excludes `CacheHitSource.DEVICE` (local HBM, never an offload tier).
    """
    return {
        source: count
        for source, count in tokens_by_source.items()
        if (
            count > 0
            and isinstance(source, CacheHitSource)
            and source != CacheHitSource.DEVICE
        )
    }


def compute_store_key_token_ranges(
    keys: Iterable[OffloadKey],
    req_context: ReqContext,
    tokens_per_chunk_by_group: Mapping[int, int],
) -> dict[OffloadKey, tuple[int, int]]:
    """Derive `[start_token, end_token)` spans for candidate store keys.

    `req_context.get_offload_key_position(key)` records the 1-based exclusive
    end token boundary of `key`; rounding `(end_token - 1)` down to a multiple
    of `tokens_per_chunk` recovers the chunk's `start_token`, handling both full
    blocks and partial-tail boundary keys.
    """
    ranges: dict[OffloadKey, tuple[int, int]] = {}
    if not tokens_per_chunk_by_group:
        return ranges
    for key in keys:
        end_token = req_context.get_offload_key_position(key)
        if end_token is None or end_token <= 0:
            continue
        group_idx = get_offload_group_idx(key)
        tokens_per_chunk = tokens_per_chunk_by_group.get(group_idx)
        if tokens_per_chunk is None or tokens_per_chunk <= 0:
            continue
        start_token = ((end_token - 1) // tokens_per_chunk) * tokens_per_chunk
        if start_token < end_token:
            ranges[key] = (start_token, end_token)
    return ranges


def count_merged_tokens(ranges: Iterable[tuple[int, int]]) -> int:
    """Count unique covered tokens across potentially overlapping intervals.

    Multi-group and hybrid models emit separate store keys per KV cache group
    for the same prompt span `[lo, hi)`; merging intervals prevents counting the
    same saved prompt tokens once per group.
    """
    intervals = sorted((lo, hi) for lo, hi in ranges if lo < hi)
    if not intervals:
        return 0
    total = 0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi:
            if hi > cur_hi:
                cur_hi = hi
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return total


@dataclass
class LoadTimingState:
    """Per-job transfer timing and source attribution state for an active load.

    A single load job can read from multiple tiers in parallel (`store.load()`
    for `HOST` DRAM and `store.read_remote()` for `P2P` RDMA + `DISK` storage),
    and Raiden reports completion per `sub_hash`. Tracking `pending_by_source`
    independently per tier lets each tier stamp its own completion latency as
    soon as its last `sub_hash` finishes, without waiting for slower tiers in
    the same job, and isolates `failed_sources` to the failing tier(s).
    """

    started_at: float
    sub_hash_source: dict[bytes, CacheHitSource]
    pending_by_source: dict[CacheHitSource, set[bytes]]
    tokens_by_source: dict[CacheHitSource, int]
    num_sub_hashes: int
    completed_at_by_source: dict[CacheHitSource, float] = field(default_factory=dict)
    failed_sources: set[CacheHitSource] = field(default_factory=set)

    def mark_launch_failed(
        self,
        failed_hashes: Iterable[bytes],
        *,
        unlaunched_hashes: Iterable[bytes] = (),
    ) -> None:
        """Record sub-hashes whose transfer launch failed synchronously."""
        for sub_hash in failed_hashes:
            source = self.sub_hash_source.get(sub_hash)
            if source is None:
                continue
            self.failed_sources.add(source)
            self.pending_by_source[source].discard(sub_hash)
            self.completed_at_by_source.pop(source, None)
        # If `store.load()` fails before `store.read_remote()` is called, drop
        # `unlaunched_hashes` from pending without blaming their remote tier.
        for sub_hash in unlaunched_hashes:
            source = self.sub_hash_source.get(sub_hash)
            if source is not None:
                self.pending_by_source[source].discard(sub_hash)

    def on_sub_hash_done(self, sub_hash: bytes, now: float) -> None:
        """Mark one sub-hash transfer done and stamp source completion if last."""
        source = self.sub_hash_source.get(sub_hash)
        if source is None:
            return
        remaining = self.pending_by_source.get(source)
        if remaining is None or sub_hash not in remaining:
            return
        remaining.discard(sub_hash)
        if not remaining and source not in self.failed_sources:
            self.completed_at_by_source[source] = now

    def on_sub_hash_failed(self, sub_hash: bytes) -> None:
        """Mark one sub-hash's source as failed on polled transfer error."""
        source = self.sub_hash_source.get(sub_hash)
        if source is None:
            return
        self.failed_sources.add(source)
        remaining = self.pending_by_source.get(source)
        if remaining is not None:
            remaining.discard(sub_hash)
        self.completed_at_by_source.pop(source, None)


@dataclass
class SaveTimingState:
    """Per-job transfer timing and token count state for an active save."""

    started_at: float
    saved_tokens: int
    num_sub_hashes: int


class RaidenOffloadStatsCollector:
    """Scheduler-side collector producing `OffloadingConnectorStats` deltas."""

    def __init__(
        self,
        *,
        bytes_per_sub_block: int = 0,
        storage_backend: str | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self._bytes_per_sub_block = bytes_per_sub_block
        self._storage_backend = storage_backend
        self._clock = clock
        self._stats = OffloadingConnectorStats()
        # Single-step map of OffloadKey -> per-sub-block CacheHitSources:
        # populated in `record_lookup_hit`, updated with pinned residency in
        # `record_prepared_load`, and cleared in `on_schedule_end()`.
        self._step_key_sources: dict[OffloadKey, tuple[CacheHitSource, ...]] = {}

    def now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return time.monotonic()

    def source_labels(self, source: CacheHitSource) -> tuple[str, str]:
        """Map an external `CacheHitSource` to `(source, backend)` labels."""
        # Reject raw `str` before `==` checks since `CacheHitSource` is a `str`.
        if not isinstance(source, CacheHitSource):
            raise ValueError(f"Unsupported offload source for metrics: {source!r}")
        if source == CacheHitSource.HOST:
            return HOST_OFFLOAD_LABELS
        if source == CacheHitSource.P2P:
            return P2P_OFFLOAD_LABELS
        if source == CacheHitSource.DISK:
            if self._storage_backend is None:
                logger.warning_once(
                    "TPURaidenOffloadingConnector: disk source observed without "
                    "configured storage_backend; using backend=%r.",
                    UNKNOWN_BACKEND_LABEL,
                )
                return (CacheHitSource.DISK.value, UNKNOWN_BACKEND_LABEL)
            return (CacheHitSource.DISK.value, self._storage_backend)
        if source == CacheHitSource.EXTERNAL_UNSPECIFIED:
            return UNSPECIFIED_OFFLOAD_LABELS
        raise ValueError(f"Unsupported offload source for metrics: {source!r}")

    def _save_destination_labels(self) -> tuple[str, str]:
        """Return `(destination, backend)` for save latency and failure metrics.

        When secondary storage is configured, Raiden `KVCacheStore::Save()`
        writes through `HBM -> Host DRAM -> Shared Storage` atomically and only
        marks `poll_save_status()` done once storage commits.
        """
        if self._storage_backend is not None:
            return (CacheHitSource.DISK.value, self._storage_backend)
        return HOST_OFFLOAD_LABELS

    def _resolve_block_source(
        self, status: Any, block_status_cls: Any
    ) -> CacheHitSource:
        source = status_to_hit_source(status, block_status_cls)
        if source == CacheHitSource.EXTERNAL_UNSPECIFIED:
            logger.warning_once(
                "TPURaidenOffloadingConnector: unrecognized Raiden BlockStatus %s; "
                "attributing offload metrics to (%r, %r).",
                repr(status),
                *UNSPECIFIED_OFFLOAD_LABELS,
            )
        return source

    def record_lookup_hit(
        self,
        key: OffloadKey,
        matches: Sequence[tuple[bytes, Any]],
        block_status_cls: Any,
    ) -> None:
        """Record per-sub-block sources that answered `lookup()` for `key`."""
        if not matches:
            return
        self._step_key_sources[key] = tuple(
            self._resolve_block_source(block.status, block_status_cls)
            for _, block in matches
        )

    def record_prepared_load(
        self,
        keys: Sequence[OffloadKey],
        device_block_size_factor: int,
        remote_slices: Mapping[bytes, Any],
        sub_hashes: Sequence[bytes],
        block_status_cls: Any,
    ) -> None:
        """Refresh per-key sub-block sources from `prepare_load` residency.

        Between `lookup()` and `prepare_load()`, a remote/disk sub-block may be
        promoted into local `HOST` DRAM; `remote_slices` contains only the
        sub-hashes that remain non-local after pinning.
        """
        for idx, key in enumerate(keys):
            start = idx * device_block_size_factor
            end = start + device_block_size_factor
            key_sources: list[CacheHitSource] = []
            for sub_hash in sub_hashes[start:end]:
                remote_block = remote_slices.get(sub_hash)
                if remote_block is None:
                    key_sources.append(CacheHitSource.HOST)
                else:
                    key_sources.append(
                        self._resolve_block_source(
                            remote_block.status, block_status_cls
                        )
                    )
            if key_sources:
                self._step_key_sources[key] = tuple(key_sources)

    def get_load_sub_sources(
        self, key: OffloadKey
    ) -> tuple[CacheHitSource, ...] | None:
        """Return recorded per-sub-block sources for `key`, or `None` on miss."""
        sub_sources = self._step_key_sources.get(key)
        if sub_sources:
            return sub_sources
        logger.warning_once(
            "TPURaidenOffloadingConnector: offload key missed step source cache; "
            "skipping offload metric attribution for unrecorded key."
        )
        return None

    def compute_load_tokens_by_source(
        self,
        *,
        start_token: int,
        end_token: int,
        load_key_ranges: Iterable[tuple[int, int, OffloadKey]],
        tokens_per_chunk_by_group: Mapping[int, int],
        mamba_group_ids: Collection[int] = (),
    ) -> dict[CacheHitSource, int]:
        """Compute deduplicated external token counts at sub-block granularity.

        Phase 1 converts each key's clipped `[lo, hi)` span into interval events:
        - Attention keys store K/V vectors per token, so `[lo, hi)` is split
          across `device_block_size_factor` sub-blocks (`tokens_per_sub` each),
          preserving exact `HOST` / `P2P` / `DISK` ratios within a mixed key.
        - Recurrent (`MambaSpec`) keys slice a single state checkpoint by bytes
          rather than token positions; all byte slices are needed together to
          restore that checkpoint, so `[lo, hi)` uses `CacheHitSource.outermost`.
        - Keys missing from `_step_key_sources` are skipped rather than
          mislabeled as `EXTERNAL_UNSPECIFIED`.

        Phase 2 sweeps sorted event boundaries across `[start_token, end_token)`
        to deduplicate overlapping ranges emitted by multi-group / hybrid models
        for the same prompt tokens, attributing shared spans via
        `CacheHitSource.outermost` (the slowest tier required to materialize the
        span).
        """
        if end_token <= start_token:
            return {}

        # Phase 1: build `(token_pos, +1/-1, source)` interval events.
        events: list[tuple[int, int, CacheHitSource]] = []
        for raw_lo, raw_hi, key in load_key_ranges:
            # Clip out tokens already resident in device HBM (`< start_token`).
            lo = max(start_token, raw_lo)
            hi = min(end_token, raw_hi)
            if lo >= hi:
                continue
            sub_sources = self.get_load_sub_sources(key)
            if sub_sources is None:
                continue
            group_idx = get_offload_group_idx(key)
            tokens_per_chunk = tokens_per_chunk_by_group.get(group_idx, 0)
            num_sub = len(sub_sources)
            if (
                group_idx in mamba_group_ids
                or num_sub <= 1
                or tokens_per_chunk <= 0
                or tokens_per_chunk % num_sub != 0
            ):
                source = CacheHitSource.outermost(sub_sources)
                events.extend(((lo, 1, source), (hi, -1, source)))
                continue

            tokens_per_sub = tokens_per_chunk // num_sub
            chunk_start = (lo // tokens_per_chunk) * tokens_per_chunk
            for sub_idx, source in enumerate(sub_sources):
                sub_lo = max(lo, chunk_start + sub_idx * tokens_per_sub)
                sub_hi = min(hi, chunk_start + (sub_idx + 1) * tokens_per_sub)
                if sub_lo < sub_hi:
                    events.extend(((sub_lo, 1, source), (sub_hi, -1, source)))

        if not events:
            return {}

        # Phase 2: sweep sorted endpoints to deduplicate overlapping KV groups.
        sources: dict[CacheHitSource, int] = {}
        active: Counter[CacheHitSource] = Counter()
        position = start_token
        for boundary, delta, source in sorted(events):
            if boundary > position:
                active_sources = [src for src, cnt in active.items() if cnt > 0]
                if active_sources:
                    resolved = CacheHitSource.outermost(active_sources)
                    sources[resolved] = sources.get(resolved, 0) + (boundary - position)
            active[source] += delta
            position = boundary
        return filter_positive_source_tokens(sources)

    def on_schedule_end(self) -> None:
        self._step_key_sources.clear()

    def record_admission_hits(
        self,
        tokens_by_source: Mapping[CacheHitSource, int],
        *,
        preempted: bool = False,
    ) -> None:
        """Increment `vllm:kv_offload_hit_tokens` for non-preempted admissions.

        Matches upstream `Scheduler.schedule()`: re-admissions after preemption
        (`request.num_preemptions > 0`) skip `HIT_TOKENS` so per-request prefix
        hits are not double-counted, whereas `LOADED_TOKENS` still counts every
        completed physical reload into HBM.
        """
        if preempted:
            return
        for source, count in filter_positive_source_tokens(tokens_by_source).items():
            self._stats.increase_counter(
                RaidenOffloadMetricName.HIT_TOKENS,
                count,
                labelvalues=self.source_labels(source),
            )

    def start_load_timing(
        self,
        *,
        local_hashes: Sequence[bytes],
        remote_hashes: Sequence[bytes],
        remote_slices: Sequence[Any],
        block_status_cls: Any,
        tokens_by_source: Mapping[CacheHitSource, int] | None = None,
    ) -> LoadTimingState:
        """Initialize per-source timing state for a load job about to launch."""
        sub_hash_source: dict[bytes, CacheHitSource] = {}
        pending_by_source: dict[CacheHitSource, set[bytes]] = {}
        for sub_hash in local_hashes:
            sub_hash_source[sub_hash] = CacheHitSource.HOST
            pending_by_source.setdefault(CacheHitSource.HOST, set()).add(sub_hash)
        for sub_hash, block in zip(remote_hashes, remote_slices):
            source = self._resolve_block_source(block.status, block_status_cls)
            sub_hash_source[sub_hash] = source
            pending_by_source.setdefault(source, set()).add(sub_hash)
        return LoadTimingState(
            started_at=self.now(),
            sub_hash_source=sub_hash_source,
            pending_by_source=pending_by_source,
            tokens_by_source=(
                filter_positive_source_tokens(tokens_by_source)
                if tokens_by_source
                else {}
            ),
            num_sub_hashes=len(local_hashes) + len(remote_hashes),
        )

    def on_load_finalized(
        self,
        timing: LoadTimingState | None,
        *,
        success: bool,
        poisoned: bool = False,
        now: float | None = None,
    ) -> None:
        """Emit load token, latency, failure, and byte/time/size metrics."""
        if timing is None:
            if not success:
                logger.warning_once(
                    "TPURaidenOffloadingConnector: load job finalized without "
                    "timing state; skipping failed_load_jobs attribution."
                )
            return
        finish_time = self.now() if now is None else now
        # Emit per-tier latencies before checking `success` so healthy tiers in
        # a partially failed job still record their transfer duration. Poisoned
        # jobs timed out in `drain_jobs` while reclaiming HBM and are excluded.
        if not poisoned:
            for source, completed_at in timing.completed_at_by_source.items():
                if source in timing.failed_sources:
                    continue
                self._stats.observe_histogram(
                    RaidenOffloadMetricName.LOAD_LATENCY,
                    max(0.0, completed_at - timing.started_at),
                    labelvalues=self.source_labels(source),
                )
        if success:
            # Credit loaded tokens and transfer bytes only when the entire job
            # succeeded; on any failure vLLM discards the blocks and recomputes.
            for source, count in timing.tokens_by_source.items():
                if count > 0:
                    self._stats.increase_counter(
                        RaidenOffloadMetricName.LOADED_TOKENS,
                        count,
                        labelvalues=self.source_labels(source),
                    )
            if self._bytes_per_sub_block > 0 and timing.num_sub_hashes > 0:
                total_bytes = timing.num_sub_hashes * self._bytes_per_sub_block
                end_time = max(
                    timing.completed_at_by_source.values(), default=finish_time
                )
                duration = max(0.0, end_time - timing.started_at)
                self._stats.increase_counter(
                    OffloadTransferMetricName.LOAD_BYTES, total_bytes
                )
                self._stats.increase_counter(
                    OffloadTransferMetricName.LOAD_TIME, duration
                )
                self._stats.observe_histogram(
                    OffloadTransferMetricName.LOAD_SIZE, total_bytes
                )
            return

        # Blame only the tier(s) whose transfers failed (or were still in flight
        # when `drain_jobs` poisoned the admission on timeout).
        failed_sources = set(timing.failed_sources)
        if poisoned:
            unfinished = {
                source
                for source, pending in timing.pending_by_source.items()
                if pending
            }
            failed_sources.update(unfinished or set(timing.sub_hash_source.values()))
        if not failed_sources:
            failed_sources.update(timing.sub_hash_source.values())
        if not failed_sources:
            logger.warning_once(
                "TPURaidenOffloadingConnector: failed load job had no recorded "
                "sub-block sources; attributing failed_load_jobs to (%r, %r).",
                *UNSPECIFIED_OFFLOAD_LABELS,
            )
            failed_sources.add(CacheHitSource.EXTERNAL_UNSPECIFIED)
        for source in failed_sources:
            self._stats.increase_counter(
                RaidenOffloadMetricName.FAILED_LOAD_JOBS,
                1,
                labelvalues=self.source_labels(source),
            )

    def record_failed_save(self) -> None:
        """Record a save job that failed before or during transfer."""
        self._stats.increase_counter(
            RaidenOffloadMetricName.FAILED_SAVE_JOBS,
            1,
            labelvalues=self._save_destination_labels(),
        )

    def start_save_timing(
        self,
        *,
        saved_keys: Iterable[OffloadKey],
        num_saved_sub_hashes: int,
        key_token_ranges: Mapping[OffloadKey, tuple[int, int]] | None = None,
    ) -> SaveTimingState:
        """Initialize timing and deduplicated token count for an active save."""
        saved_tokens = 0
        if key_token_ranges:
            saved_tokens = count_merged_tokens(
                key_token_ranges[key] for key in saved_keys if key in key_token_ranges
            )
        return SaveTimingState(
            started_at=self.now(),
            saved_tokens=saved_tokens,
            num_sub_hashes=num_saved_sub_hashes,
        )

    def on_save_finalized(
        self,
        timing: SaveTimingState | None,
        *,
        success: bool,
        now: float | None = None,
    ) -> None:
        """Emit save token, latency, failure, and byte/time/size metrics."""
        if not success:
            self.record_failed_save()
            return
        if timing is None:
            return
        finish_time = self.now() if now is None else now
        duration = max(0.0, finish_time - timing.started_at)
        destination_labels = self._save_destination_labels()
        self._stats.observe_histogram(
            RaidenOffloadMetricName.SAVE_LATENCY,
            duration,
            labelvalues=destination_labels,
        )
        if timing.saved_tokens > 0:
            # Write-through storage saves retain staged blocks in Host DRAM in
            # addition to writing shared storage, so credit both destinations.
            self._stats.increase_counter(
                RaidenOffloadMetricName.SAVED_TOKENS,
                timing.saved_tokens,
                labelvalues=HOST_OFFLOAD_LABELS,
            )
            if self._storage_backend is not None:
                self._stats.increase_counter(
                    RaidenOffloadMetricName.SAVED_TOKENS,
                    timing.saved_tokens,
                    labelvalues=(CacheHitSource.DISK.value, self._storage_backend),
                )
        if self._bytes_per_sub_block > 0 and timing.num_sub_hashes > 0:
            total_bytes = timing.num_sub_hashes * self._bytes_per_sub_block
            self._stats.increase_counter(
                OffloadTransferMetricName.STORE_BYTES, total_bytes
            )
            self._stats.increase_counter(OffloadTransferMetricName.STORE_TIME, duration)
            self._stats.observe_histogram(
                OffloadTransferMetricName.STORE_SIZE, total_bytes
            )

    def pop_stats(self) -> OffloadingConnectorStats | None:
        """Drain accumulated per-step stats deltas for IPC, or `None` if idle.

        Resets `self._stats` so each delta is shipped to `RaidenOffloadPromMetrics`
        in the API server process at most once.
        """
        if self._stats.is_empty():
            return None
        stats = self._stats
        self._stats = OffloadingConnectorStats()
        return stats
