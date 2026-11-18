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

- `compute_load_tokens_by_source` runs in `EngineCore`, computing deduplicated
  token counts per tier for one offload hit.
- `RaidenOffloadStatsCollector` runs in `EngineCore`, counting hit tokens,
  loaded tokens, load latencies, and failed load jobs per tier, and draining
  per-step deltas as `OffloadingConnectorStats` via `pop_stats()`.
- `RaidenOffloadPromMetrics` runs in the API server process, registering
  Raiden's `OffloadingSpec` metric definitions and applying IPC stats deltas to
  Prometheus counters and histograms.
"""

from __future__ import annotations

import copy
import enum
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass

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
    get_offload_group_idx,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


# `source` labels where data is read from, aligned with
# upstream `vllm:prompt_tokens_cached_by_source` for PromQL rollups. `backend`
# names what serves the tier: `dram` for host, `rdma` for p2p, and the
# configured storage type for disk, so a new storage backend adds label values
# rather than metrics.
SOURCE_LABELNAMES: tuple[str, str] = ("source", "backend")
UNKNOWN_BACKEND_LABEL = "unknown"
HOST_OFFLOAD_LABELS: tuple[str, str] = (CacheHitSource.HOST.value, "dram")
P2P_OFFLOAD_LABELS: tuple[str, str] = (CacheHitSource.P2P.value, "rdma")


def disk_offload_labels(storage_backend: str) -> tuple[str, str]:
    """Labels of the storage tier, for example `("disk", "posix")`."""
    return (CacheHitSource.DISK.value, storage_backend)


# Tier of each sub-block of one `OffloadKey`, as `(source, count)` runs of
# consecutive sub-blocks in the same tier; the counts sum to
# `device_block_size_factor`. For example, with 4 sub-blocks per key,
# `((HOST, 3), (DISK, 1))` means sub-blocks 0-2 are in host DRAM and 3 is on
# disk, and a key entirely in host DRAM is `((HOST, 4),)`.
SubBlockSourceCounts = tuple[tuple[CacheHitSource, int], ...]

# Completion is observed once per scheduler step, so the sub-millisecond buckets
# only fill when steps run faster than the transfer (an otherwise idle engine);
# the upper buckets cover peer and storage reads under load.
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
    """Prometheus metrics emitted by Raiden KV offload."""

    HIT_TOKENS = "vllm:kv_offload_raiden_hit_tokens"
    LOADED_TOKENS = "vllm:kv_offload_raiden_loaded_tokens"
    LOAD_OBSERVED_LATENCY = "vllm:kv_offload_raiden_load_observed_latency_seconds"
    FAILED_LOAD_JOBS = "vllm:kv_offload_raiden_failed_load_jobs"


def build_raiden_offload_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    """Return Prometheus metric definitions for TPURaidenStoreOffloadingSpec."""
    return {
        RaidenOffloadMetricName.HIT_TOKENS: OffloadingCounterMetadata(
            documentation=(
                "Number of prompt tokens matched in external KV offload tiers, "
                "labeled by source and backend. Counted at scheduler admission, "
                "for each load that prepare_load accepted. Unlike "
                "vllm:prompt_tokens_cached_by_source, splits disk hits by storage "
                "backend and leaves out hits whose load was not accepted."
            ),
            labelnames=SOURCE_LABELNAMES,
        ),
        RaidenOffloadMetricName.LOADED_TOKENS: OffloadingCounterMetadata(
            documentation=(
                "Number of KV cache tokens successfully loaded into device HBM "
                "from external offload tiers, labeled by source and backend."
            ),
            labelnames=SOURCE_LABELNAMES,
        ),
        RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY: OffloadingHistogramMetadata(
            documentation=(
                "Seconds from launching a KV offload load until the scheduler "
                "observes each tier's last sub-block land, labeled by source "
                "and backend. Completion is polled once per scheduler step, so "
                "values are rounded up to the next step (at least one forward "
                "pass while a batch runs)."
            ),
            labelnames=SOURCE_LABELNAMES,
            buckets=OFFLOAD_LATENCY_BUCKETS,
        ),
        RaidenOffloadMetricName.FAILED_LOAD_JOBS: OffloadingCounterMetadata(
            documentation=(
                "Number of KV offload load jobs that failed after prepare_load "
                "accepted them, labeled by the source and backend of each tier "
                "that failed."
            ),
            labelnames=SOURCE_LABELNAMES,
        ),
    }


def _add_unobserved_histogram(
    stats: OffloadingConnectorStats, name: str, labelvalues: tuple[str, ...]
) -> None:
    """Add the `name` histogram series of `labelvalues` to `stats`, with no sample.

    `OffloadingConnectorStats` adds a histogram series only together with a
    sample, so record one and drop it.
    """
    stats.observe_histogram(name, 0.0, labelvalues=labelvalues)
    _, values = stats.metric_sections()
    values[name][labelvalues].clear()


class RaidenOffloadPromMetrics(OffloadPromMetrics):
    """Prometheus metrics binder for TPURaidenOffloadingConnector.

    `TPURaidenOffloadingConnector` instantiates `TPURaidenStoreOffloadingSpec`
    directly (bypassing `OffloadingSpecFactory`), whereas upstream
    `OffloadPromMetrics.__init__` in the API server process resolves
    `spec_cls.build_metric_definitions` via `OffloadingSpecFactory` using
    `kv_connector_extra_config["spec_name"]`. This subclass names `spec_cls` as
    an out-of-tree spec (`spec_name` + `spec_module_path`) on shallow config
    copies, so neither the factory registry nor the caller's `VllmConfig` and
    `KVTransferConfig` are mutated.

    `configured_tier_labels` holds the `(source, backend)` labels of the tiers
    the deployment enables. Each engine's hit and load series for these tiers
    are created at 0 on startup. Prometheus counts a series from the first
    value it scrapes, so `rate()` and `increase()` would miss the first
    increment of a series that only appeared with that increment.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        *,
        spec_cls: type[OffloadingSpec],
        configured_tier_labels: Collection[tuple[str, str]],
    ):
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        scoped_transfer_config = copy.copy(kv_transfer_config)
        scoped_transfer_config.kv_connector_extra_config = {
            **kv_transfer_config.kv_connector_extra_config,
            "spec_name": spec_cls.__name__,
            "spec_module_path": spec_cls.__module__,
        }
        scoped_vllm_config = copy.copy(vllm_config)
        scoped_vllm_config.kv_transfer_config = scoped_transfer_config
        super().__init__(
            scoped_vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )
        zero_stats = OffloadingConnectorStats()
        for labelvalues in configured_tier_labels:
            for counter in (
                RaidenOffloadMetricName.HIT_TOKENS,
                RaidenOffloadMetricName.LOADED_TOKENS,
                RaidenOffloadMetricName.FAILED_LOAD_JOBS,
            ):
                zero_stats.increase_counter(counter, 0, labelvalues=labelvalues)
            _add_unobserved_histogram(
                zero_stats, RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY, labelvalues
            )
        for engine_idx in per_engine_labelvalues:
            self.observe(zero_stats.data, engine_idx)


def external_sources_with_tokens(
    tokens_by_source: Mapping[CacheHitSource, int],
) -> dict[CacheHitSource, int]:
    """Keep the external sources that have at least one token.

    For example, `{DEVICE: 16, HOST: 64, P2P: 0, DISK: 128}` becomes
    `{HOST: 64, DISK: 128}`: `DEVICE` is local HBM, never an offload tier.
    Checks `isinstance(source, CacheHitSource)` explicitly because
    `CacheHitSource` inherits from `str` (`"host" == CacheHitSource.HOST`).
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


def split_external_hit(
    tokens_by_source: Mapping[CacheHitSource, int], num_external_tokens: int
) -> dict[CacheHitSource, int]:
    """Complete a per-tier hit split so it sums to `num_external_tokens`.

    Upstream `PrefixCacheStats.record` replaces any split that does not sum to
    the hit count with all-`external_unspecified`. Tokens no tier supplied
    (spans no loaded key covers, such as omitted sliding-window history) are
    reported as `EXTERNAL_UNSPECIFIED` so the known tiers survive. For example,
    `split_external_hit({HOST: 768, DISK: 512}, 1536)` returns
    `{HOST: 768, DISK: 512, EXTERNAL_UNSPECIFIED: 256}`; the incomplete split
    would have been reported as `{EXTERNAL_UNSPECIFIED: 1536}`.
    """
    split = dict(tokens_by_source)
    unattributed = num_external_tokens - sum(split.values())
    if unattributed > 0:
        unspecified = CacheHitSource.EXTERNAL_UNSPECIFIED
        split[unspecified] = split.get(unspecified, 0) + unattributed
    return split


def compute_load_tokens_by_source(
    *,
    start_token: int,
    end_token: int,
    load_key_ranges: Iterable[tuple[int, int, OffloadKey]],
    key_sources: Mapping[OffloadKey, SubBlockSourceCounts],
    tokens_per_chunk_by_group: Mapping[int, int],
    mamba_group_ids: Collection[int] = (),
) -> dict[CacheHitSource, int]:
    """Compute deduplicated external token counts at sub-block granularity.

    Phase 1 converts each key's clipped `[lo, hi)` span into interval events
    from its `SubBlockSourceCounts` in `key_sources`:
    - Uniform keys (`len(source_counts) == 1`) emit a single `[lo, hi)`
      interval in O(1) regardless of `device_block_size_factor`.
    - Mixed Attention keys split `[lo, hi)` across `(source, count)` pairs
      (`count * tokens_per_sub` tokens each), preserving exact `HOST` /
      `P2P` / `DISK` ratios without iterating individual sub-blocks.
    - Recurrent (`MambaSpec`) keys slice a single state checkpoint by bytes
      rather than token positions; all byte slices are needed together to
      restore that checkpoint, so `[lo, hi)` uses `CacheHitSource.outermost`.
    - A mixed key that cannot be split evenly by sub-block also uses
      `CacheHitSource.outermost`, and logs a warning.

    Phase 2 sweeps sorted event boundaries across `[start_token, end_token)`
    to deduplicate overlapping ranges emitted by multi-group / hybrid models
    for the same prompt tokens, attributing shared spans via
    `CacheHitSource.outermost` (the slowest tier required to materialize the
    span). Tokens no key covers are left out; `split_external_hit`
    reports them to upstream as `EXTERNAL_UNSPECIFIED`.

    For example, in a hybrid model whose attention keys for tokens
    `[0, 1536)` are in host DRAM and whose Mamba key for `[768, 1536)` is on
    disk, `[768, 1536)` needs both tiers and counts as the slower one:
    `{HOST: 768, DISK: 768}`.
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
        source_counts = key_sources[key]
        if len(source_counts) == 1:
            only_source = source_counts[0][0]
            events.extend(((lo, 1, only_source), (hi, -1, only_source)))
            continue

        group_idx = get_offload_group_idx(key)
        is_mamba = group_idx in mamba_group_ids
        tokens_per_chunk = tokens_per_chunk_by_group.get(group_idx, 0)
        num_sub = sum(count for _, count in source_counts)
        if is_mamba or tokens_per_chunk <= 0 or tokens_per_chunk % num_sub != 0:
            if not is_mamba:
                logger.warning_once(
                    "TPURaidenOffloadingConnector: cannot split a mixed-tier "
                    "key of KV cache group %d by sub-block (%d tokens per "
                    "chunk, %d sub-blocks); attributing it to its slowest tier.",
                    group_idx,
                    tokens_per_chunk,
                    num_sub,
                )
            source = CacheHitSource.outermost(src for src, _ in source_counts)
            events.extend(((lo, 1, source), (hi, -1, source)))
            continue

        tokens_per_sub = tokens_per_chunk // num_sub
        chunk_start = (lo // tokens_per_chunk) * tokens_per_chunk
        sub_offset = 0
        for source, count in source_counts:
            span_lo = max(lo, chunk_start + sub_offset * tokens_per_sub)
            sub_offset += count
            span_hi = min(hi, chunk_start + sub_offset * tokens_per_sub)
            if span_lo < span_hi:
                events.extend(((span_lo, 1, source), (span_hi, -1, source)))

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
    return external_sources_with_tokens(sources)


@dataclass(slots=True)
class _TierProgress:
    """Completion state of the sub-hashes one load job reads from one tier."""

    source: CacheHitSource
    left: int = 0
    completed_at: float | None = None
    failed: bool = False


@dataclass
class LoadTimingState:
    """Per-tier completion and failure state for one active load job.

    A load reads from several tiers in parallel (`store.load()` for host DRAM,
    `store.read_remote()` for peers and storage), and Raiden reports completion
    per sub-hash. Each tier stamps its completion when its last sub-hash lands,
    without waiting for slower tiers, and a failure is blamed only on the
    tier(s) that failed.

    Callers report each landed sub-hash at most once, while it is still pending
    in its `Admission`, so a tier counts the sub-hashes that have not landed
    rather than duplicating the admission's pending set.
    """

    started_at: float
    host: _TierProgress
    # `host`, then each tier that the job reads via `read_remote()`.
    tiers: tuple[_TierProgress, ...]
    # Tier of each sub-hash read via `read_remote()`; every other sub-hash is a
    # host DRAM load.
    remote: dict[bytes, _TierProgress]
    tokens_by_source: dict[CacheHitSource, int]

    def mark_local_launch_failed(self) -> None:
        """`store.load()` refused the job's host DRAM batch.

        `read_remote()` is then never called, so the remote tiers are not
        blamed.
        """
        self.host.failed = True

    def mark_remote_launch_failed(self, sub_hashes: Iterable[bytes]) -> None:
        """`store.read_remote()` refused a batch of peer and storage reads.

        `sub_hashes` are the refused batch plus every batch not yet launched;
        their tiers are blamed, while batches already in flight report through
        the poll as usual.
        """
        for sub_hash in sub_hashes:
            self.remote[sub_hash].failed = True

    def mark_poisoned(self) -> None:
        """Blame the tiers still in flight when `drain_jobs` timed out."""
        for tier in self.tiers:
            if tier.left:
                tier.failed = True

    def on_sub_hash_done(self, sub_hash: bytes, now: float) -> None:
        """Count one landed sub-hash; stamp its tier's completion if last."""
        tier = self.remote.get(sub_hash, self.host)
        tier.left -= 1
        if not tier.left and not tier.failed:
            tier.completed_at = now

    def on_sub_hash_failed(self, sub_hash: bytes) -> None:
        """Blame the tier of one failed sub-hash."""
        tier = self.remote.get(sub_hash, self.host)
        tier.failed = True
        tier.completed_at = None


class RaidenOffloadStatsCollector:
    """Scheduler-side collector producing `OffloadingConnectorStats` deltas."""

    def __init__(
        self,
        *,
        storage_backend: str | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self._storage_backend = storage_backend
        self._clock = clock
        self._stats = OffloadingConnectorStats()

    def now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return time.monotonic()

    def source_labels(self, source: CacheHitSource) -> tuple[str, str]:
        """Map an external `CacheHitSource` to `(source, backend)` labels.

        For example, `HOST` maps to `("host", "dram")`, `P2P` to
        `("p2p", "rdma")`, and `DISK` to `("disk", "posix")` with a POSIX
        storage tier.

        Assumes one storage backend per engine: Raiden's `SHARED_STORAGE`
        status does not say which backend holds a block. If an engine read from
        two storage backends, hits from both would get the `storage_backend`
        label until Raiden reports the backend per block.
        """
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
                return disk_offload_labels(UNKNOWN_BACKEND_LABEL)
            return disk_offload_labels(self._storage_backend)
        raise ValueError(f"Unsupported offload source for metrics: {source!r}")

    def record_admission_hits(
        self,
        tokens_by_source: Mapping[CacheHitSource, int],
        *,
        preempted: bool = False,
    ) -> None:
        """Increment `vllm:kv_offload_raiden_hit_tokens` for non-preempted admissions.

        Matches upstream `Scheduler.schedule()`: re-admissions after preemption
        (`request.num_preemptions > 0`) skip `HIT_TOKENS` so per-request prefix
        hits are not double-counted, whereas `LOADED_TOKENS` still counts every
        completed physical reload into HBM.
        """
        if preempted:
            return
        for source, count in external_sources_with_tokens(tokens_by_source).items():
            self._stats.increase_counter(
                RaidenOffloadMetricName.HIT_TOKENS,
                count,
                labelvalues=self.source_labels(source),
            )

    def start_load_timing(
        self,
        *,
        num_host_sub_hashes: int,
        remote_sources: Mapping[bytes, CacheHitSource],
        tokens_by_source: Mapping[CacheHitSource, int] | None = None,
    ) -> LoadTimingState:
        """Initialize per-tier timing state for a load job about to launch.

        `remote_sources` has the tier of each sub-hash that the job reads via
        `read_remote()`: `P2P` or `DISK`. The job's other
        `num_host_sub_hashes` sub-hashes are host DRAM loads.
        """
        host = _TierProgress(CacheHitSource.HOST, left=num_host_sub_hashes)
        tiers = {CacheHitSource.HOST: host}
        remote: dict[bytes, _TierProgress] = {}
        for sub_hash, source in remote_sources.items():
            tier = tiers.get(source)
            if tier is None:
                tier = tiers[source] = _TierProgress(source)
            tier.left += 1
            remote[sub_hash] = tier
        return LoadTimingState(
            started_at=self.now(),
            host=host,
            tiers=tuple(tiers.values()),
            remote=remote,
            tokens_by_source=external_sources_with_tokens(tokens_by_source or {}),
        )

    def on_load_finalized(
        self,
        timing: LoadTimingState,
        *,
        success: bool,
        poisoned: bool = False,
    ) -> None:
        """Emit load token, latency, and failure metrics."""
        tiers = timing.tiers
        completed = [
            (tier.source, completed_at)
            for tier in tiers
            if (completed_at := tier.completed_at) is not None
        ]
        # Emit per-tier latencies before checking `success` so healthy tiers in
        # a partially failed job still record their transfer duration. Failed
        # tiers never stamp a completion. Poisoned jobs timed out in
        # `drain_jobs` while reclaiming HBM and are excluded.
        if not poisoned:
            for source, completed_at in completed:
                self._stats.observe_histogram(
                    RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY,
                    max(0.0, completed_at - timing.started_at),
                    labelvalues=self.source_labels(source),
                )
        if success:
            # Credit loaded tokens only when the entire job succeeded; on any
            # failure vLLM discards the blocks and recomputes.
            for source, count in timing.tokens_by_source.items():
                self._stats.increase_counter(
                    RaidenOffloadMetricName.LOADED_TOKENS,
                    count,
                    labelvalues=self.source_labels(source),
                )
            return

        # Blame only the tier(s) whose transfers failed, or that were still in
        # flight when `drain_jobs` poisoned the job (`mark_poisoned`). Every
        # failed job has such a tier: each path that fails a load marks one.
        for tier in tiers:
            if tier.failed:
                self._stats.increase_counter(
                    RaidenOffloadMetricName.FAILED_LOAD_JOBS,
                    1,
                    labelvalues=self.source_labels(tier.source),
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
