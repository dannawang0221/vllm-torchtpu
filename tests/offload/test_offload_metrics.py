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
"""Unit tests for Raiden KV offload Prometheus metrics and stats collectors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import prometheus_client
import pytest
import torch
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import PromMetric
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    get_connector_metric_definitions,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector
from vllm_torchtpu.offload.metrics import (
    DESTINATION_LABELNAMES,
    SOURCE_LABELNAMES,
    OffloadTransferMetricName,
    RaidenOffloadMetricName,
    RaidenOffloadPromMetrics,
    RaidenOffloadStatsCollector,
    StoreKeyTokenRanges,
    SubBlockSourceCounts,
    build_raiden_offload_metric_definitions,
    compute_load_tokens_by_source,
    compute_store_key_token_ranges,
    count_merged_tokens,
    external_sources_with_tokens,
    saved_sub_block_ranges,
    select_saved_tokens_group,
    split_external_hit,
)
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

_HOST = ("host", "dram")
_P2P = ("p2p", "rdma")
_DISK = ("disk", "posix")
_SUB_BLOCK_BYTES = 256

_ATTN = FullAttentionSpec(
    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
)
_SWA = SlidingWindowSpec(
    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, sliding_window=64
)
_MAMBA = MambaSpec(
    block_size=16, shapes=((2, 8),), dtypes=(torch.bfloat16,), mamba_cache_mode="align"
)


def _group(group_idx: int, kv_cache_spec: Any) -> Any:
    """The fields of vLLM's `GroupOffloadConfig` that the save metrics read."""
    return SimpleNamespace(
        group_idx=group_idx, tokens_per_chunk=16, kv_cache_spec=kv_cache_spec
    )


def _collector(**kwargs: Any) -> RaidenOffloadStatsCollector:
    return RaidenOffloadStatsCollector(bytes_per_sub_block=_SUB_BLOCK_BYTES, **kwargs)


def _pop_values(collector: RaidenOffloadStatsCollector) -> dict[str, Any]:
    stats = collector.pop_stats()
    assert stats is not None
    return stats.data["data"]


def _make_prom_metric_types(
    registry: prometheus_client.CollectorRegistry,
) -> dict[type[PromMetric], type[PromMetric]]:
    class _Counter(prometheus_client.Counter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, registry=registry, **kwargs)

    class _Gauge(prometheus_client.Gauge):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, registry=registry, **kwargs)

    class _Histogram(prometheus_client.Histogram):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, registry=registry, **kwargs)

    return {
        prometheus_client.Counter: _Counter,
        prometheus_client.Gauge: _Gauge,
        prometheus_client.Histogram: _Histogram,
    }


def test_transfer_metric_names_match_upstream_connector_definitions() -> None:
    upstream_defs = get_connector_metric_definitions()
    for metric_name in OffloadTransferMetricName:
        assert metric_name.value in upstream_defs


def test_metric_definitions_have_source_or_destination_labels() -> None:
    defs = build_raiden_offload_metric_definitions()
    assert set(defs.keys()) == {m.value for m in RaidenOffloadMetricName}
    for metric in (
        RaidenOffloadMetricName.HIT_TOKENS,
        RaidenOffloadMetricName.LOADED_TOKENS,
        RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY,
        RaidenOffloadMetricName.FAILED_LOAD_JOBS,
    ):
        assert defs[metric].labelnames == SOURCE_LABELNAMES
    for metric in (
        RaidenOffloadMetricName.SAVED_TOKENS,
        RaidenOffloadMetricName.SAVE_OBSERVED_LATENCY,
        RaidenOffloadMetricName.FAILED_SAVE_JOBS,
    ):
        assert defs[metric].labelnames == DESTINATION_LABELNAMES


def test_external_sources_with_tokens() -> None:
    raw = {
        CacheHitSource.DEVICE: 16,
        CacheHitSource.HOST: 64,
        CacheHitSource.P2P: 0,
        CacheHitSource.DISK: 128,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 32,
    }
    assert external_sources_with_tokens(raw) == {
        CacheHitSource.HOST: 64,
        CacheHitSource.DISK: 128,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 32,
    }


def test_collector_labels_disk_without_storage_backend_as_unknown() -> None:
    collector = _collector(storage_backend=None)
    assert collector.source_labels(CacheHitSource.DISK) == ("disk", "unknown")


@pytest.mark.parametrize(
    "source",
    [
        CacheHitSource.DEVICE,
        CacheHitSource.EXTERNAL_UNSPECIFIED,
        pytest.param("host", id="raw str"),
    ],
)
def test_collector_rejects_unsupported_sources(source: Any) -> None:
    collector = _collector(storage_backend=None)
    with pytest.raises(ValueError, match="Unsupported offload source"):
        collector.source_labels(source)


_KEY_0 = make_offload_key(b"key0", 0)
_KEY_1 = make_offload_key(b"key1", 0)


@pytest.mark.parametrize(
    ("end_token", "tokens_per_chunk", "load_key_ranges", "key_sources", "expected"),
    [
        # A 64-token key of four 16-token sub-blocks: 3 HOST + 1 DISK sub-blocks
        # form 2 runs -> 48 HOST + 16 DISK.
        pytest.param(
            64,
            64,
            [(0, 64, _KEY_0)],
            {_KEY_0: ((CacheHitSource.HOST, 3), (CacheHitSource.DISK, 1))},
            {CacheHitSource.HOST: 48, CacheHitSource.DISK: 16},
            id="one key in two tiers",
        ),
        # The range [0, 48) of a 64-token chunk excludes its last sub-block,
        # [48, 64).
        pytest.param(
            48,
            64,
            [(0, 48, _KEY_0)],
            {
                _KEY_0: (
                    (CacheHitSource.HOST, 2),
                    (CacheHitSource.P2P, 1),
                    (CacheHitSource.DISK, 1),
                )
            },
            {CacheHitSource.HOST: 32, CacheHitSource.P2P: 16},
            id="partial tail",
        ),
        # 8192-token chunks of 512 sub-blocks: a uniform HOST key, and a key
        # of 384 HOST + 128 DISK sub-blocks in 2 runs.
        pytest.param(
            16384,
            8192,
            [(0, 8192, _KEY_0), (8192, 16384, _KEY_1)],
            {
                _KEY_0: ((CacheHitSource.HOST, 512),),
                _KEY_1: ((CacheHitSource.HOST, 384), (CacheHitSource.DISK, 128)),
            },
            {CacheHitSource.HOST: 8192 + 6144, CacheHitSource.DISK: 2048},
            id="512 sub-blocks per key",
        ),
    ],
)
def test_compute_load_tokens_by_source_splits_mixed_key_by_sub_block(
    end_token: int,
    tokens_per_chunk: int,
    load_key_ranges: list[tuple[int, int, OffloadKey]],
    key_sources: dict[OffloadKey, SubBlockSourceCounts],
    expected: dict[CacheHitSource, int],
) -> None:
    assert (
        compute_load_tokens_by_source(
            start_token=0,
            end_token=end_token,
            load_key_ranges=load_key_ranges,
            key_sources=key_sources,
            tokens_per_chunk_by_group={0: tokens_per_chunk},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("mamba_sources", "expected"),
    [
        pytest.param(
            ((CacheHitSource.HOST, 4),),
            {CacheHitSource.HOST: 24, CacheHitSource.P2P: 8},
            id="recurrent state in host DRAM",
        ),
        # Mamba's indivisible checkpoint has one byte slice in DISK, so the
        # whole prefix [0, 32) depends on that DISK checkpoint.
        pytest.param(
            ((CacheHitSource.HOST, 3), (CacheHitSource.DISK, 1)),
            {CacheHitSource.DISK: 32},
            id="recurrent state partly on disk",
        ),
    ],
)
def test_compute_load_tokens_by_source_hybrid_overlap_uses_slowest_tier(
    mamba_sources: SubBlockSourceCounts,
    expected: dict[CacheHitSource, int],
) -> None:
    # Hybrid Attention (group 0) + Mamba (group 1) over [0, 32):
    # k_a0=(HOST*4), k_a1=(HOST*2, P2P*2), and k_m1 covers the whole prefix.
    k_a0 = make_offload_key(b"a0", 0)
    k_a1 = make_offload_key(b"a1", 0)
    k_m1 = make_offload_key(b"m1", 1)
    assert (
        compute_load_tokens_by_source(
            start_token=0,
            end_token=32,
            load_key_ranges=[(0, 16, k_a0), (16, 32, k_a1), (0, 32, k_m1)],
            key_sources={
                k_a0: ((CacheHitSource.HOST, 4),),
                k_a1: ((CacheHitSource.HOST, 2), (CacheHitSource.P2P, 2)),
                k_m1: mamba_sources,
            },
            tokens_per_chunk_by_group={0: 16, 1: 16},
            mamba_group_ids={1},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("start_token", "end_token", "key_range", "expected"),
    [
        # The HBM hit ends inside the chunk, so the load starts mid-chunk.
        (16, 64, (16, 64), {CacheHitSource.HOST: 16, CacheHitSource.DISK: 32}),
        (16, 64, (0, 64), {CacheHitSource.HOST: 16, CacheHitSource.DISK: 32}),
        # The hit starts and ends inside sub-blocks.
        (8, 40, (0, 64), {CacheHitSource.HOST: 24, CacheHitSource.DISK: 8}),
        (40, 64, (0, 64), {CacheHitSource.DISK: 24}),
        # The second chunk of a prompt: tokens [64, 128).
        (80, 128, (64, 128), {CacheHitSource.HOST: 16, CacheHitSource.DISK: 32}),
    ],
)
def test_compute_load_tokens_by_source_clips_mixed_key_to_hit_range(
    start_token: int,
    end_token: int,
    key_range: tuple[int, int],
    expected: dict[CacheHitSource, int],
) -> None:
    # A 64-token key with four 16-token sub-blocks: two in host DRAM, two on disk.
    key = make_offload_key(b"mixed", 0)
    assert (
        compute_load_tokens_by_source(
            start_token=start_token,
            end_token=end_token,
            load_key_ranges=[(*key_range, key)],
            key_sources={key: ((CacheHitSource.HOST, 2), (CacheHitSource.DISK, 2))},
            tokens_per_chunk_by_group={0: 64},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("tokens_per_chunk_by_group", "mamba_group_ids", "num_warnings"),
    [
        pytest.param({}, (), 1, id="group without a chunk size"),
        pytest.param({0: 62}, (), 1, id="chunk size not divisible by sub-blocks"),
        pytest.param({0: 64}, (0,), 0, id="mamba checkpoint"),
    ],
)
def test_compute_load_tokens_by_source_unsplit_key_uses_slowest_tier(
    tokens_per_chunk_by_group: dict[int, int],
    mamba_group_ids: tuple[int, ...],
    num_warnings: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "vllm_torchtpu.offload.metrics.logger.warning_once",
        lambda *args: warnings.append(args),
    )
    key = make_offload_key(b"mixed", 0)
    assert compute_load_tokens_by_source(
        start_token=0,
        end_token=64,
        load_key_ranges=[(0, 64, key)],
        key_sources={key: ((CacheHitSource.HOST, 3), (CacheHitSource.DISK, 1))},
        tokens_per_chunk_by_group=tokens_per_chunk_by_group,
        mamba_group_ids=mamba_group_ids,
    ) == {CacheHitSource.DISK: 64}
    assert len(warnings) == num_warnings


def test_split_external_hit_sums_to_hit_count() -> None:
    assert split_external_hit({CacheHitSource.HOST: 32}, 48) == {
        CacheHitSource.HOST: 32,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 16,
    }
    assert split_external_hit(
        {CacheHitSource.DISK: 8, CacheHitSource.EXTERNAL_UNSPECIFIED: 8}, 32
    ) == {CacheHitSource.DISK: 8, CacheHitSource.EXTERNAL_UNSPECIFIED: 24}
    assert split_external_hit({CacheHitSource.P2P: 16}, 16) == {CacheHitSource.P2P: 16}
    assert split_external_hit({}, 16) == {CacheHitSource.EXTERNAL_UNSPECIFIED: 16}


def test_load_records_loaded_tokens_and_each_tier_latency() -> None:
    clock = Mock(return_value=10.0)
    collector = _collector(storage_backend="posix", clock=clock)
    timing = collector.start_load_timing(
        num_host_sub_hashes=2,
        remote_sources={b"p0": CacheHitSource.P2P, b"d0": CacheHitSource.DISK},
        tokens_by_source={
            CacheHitSource.HOST: 16,
            CacheHitSource.P2P: 32,
            CacheHitSource.DISK: 48,
        },
    )
    # The host tier lands 2 ms after the launch, the peer read 8 ms after it,
    # and the storage read 40 ms after it.
    clock.return_value += 0.002
    timing.on_sub_hash_done(b"h0", clock())
    timing.on_sub_hash_done(b"h1", clock())
    clock.return_value += 0.006
    timing.on_sub_hash_done(b"p0", clock())
    clock.return_value += 0.032
    timing.on_sub_hash_done(b"d0", clock())
    collector.on_load_finalized(timing, success=True)

    values = _pop_values(collector)
    assert values[RaidenOffloadMetricName.LOADED_TOKENS] == {
        _HOST: 16,
        _P2P: 32,
        _DISK: 48,
    }
    assert values[RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY] == {
        _HOST: [pytest.approx(0.002)],
        _P2P: [pytest.approx(0.008)],
        _DISK: [pytest.approx(0.04)],
    }
    assert RaidenOffloadMetricName.FAILED_LOAD_JOBS not in values
    # The load as a whole: the bytes of its 4 sub-blocks, and the time of its
    # slowest tier.
    assert values[OffloadTransferMetricName.LOAD_BYTES] == {(): 4 * _SUB_BLOCK_BYTES}
    assert values[OffloadTransferMetricName.LOAD_SIZE] == {(): [4 * _SUB_BLOCK_BYTES]}
    assert values[OffloadTransferMetricName.LOAD_TIME] == {(): pytest.approx(0.04)}


@pytest.mark.parametrize(
    ("steps", "poisoned", "blamed", "timed"),
    [
        # read_remote() is never called, so the remote tiers are not blamed.
        pytest.param(
            [("mark_local_launch_failed",)],
            False,
            {_HOST},
            set(),
            id="host load refused",
        ),
        pytest.param(
            [("mark_remote_launch_failed", [b"d0"])],
            False,
            {_DISK},
            set(),
            id="storage read refused",
        ),
        # The first read_remote() batch, peer p0 and the storage read, is in
        # flight when the batch of peer p1 is refused.
        pytest.param(
            [
                ("mark_remote_launch_failed", [b"p1"]),
                ("on_sub_hash_done", b"p0", 1.0),
                ("on_sub_hash_done", b"d0", 1.0),
            ],
            False,
            {_P2P},
            {_DISK},
            id="second peer batch refused",
        ),
        pytest.param(
            [("on_sub_hash_done", b"h0", 1.0), ("on_sub_hash_failed", b"d0")],
            False,
            {_DISK},
            {_HOST},
            id="storage read failed",
        ),
        # A failure that Raiden reports for a sub-hash that had landed.
        pytest.param(
            [("on_sub_hash_done", b"h0", 1.0), ("on_sub_hash_failed", b"h0")],
            False,
            {_HOST},
            set(),
            id="failure reported after completion",
        ),
        # The host tier landed before drain_jobs timed out. A poisoned job
        # records no latency.
        pytest.param(
            [
                ("on_sub_hash_done", b"h0", 1.0),
                ("mark_poisoned",),
                ("on_sub_hash_done", b"d0", 2.0),
            ],
            True,
            {_P2P, _DISK},
            set(),
            id="poisoned with remote reads in flight",
        ),
    ],
)
def test_failed_load_blames_only_failed_or_in_flight_tiers(
    steps: list[tuple[Any, ...]],
    poisoned: bool,
    blamed: set[tuple[str, str]],
    timed: set[tuple[str, str]],
) -> None:
    collector = _collector(storage_backend="posix")
    # One host DRAM sub-hash (h0), two peer reads and one storage read.
    timing = collector.start_load_timing(
        num_host_sub_hashes=1,
        remote_sources={
            b"p0": CacheHitSource.P2P,
            b"p1": CacheHitSource.P2P,
            b"d0": CacheHitSource.DISK,
        },
        tokens_by_source={CacheHitSource.HOST: 16},
    )
    for method, *args in steps:
        getattr(timing, method)(*args)
    collector.on_load_finalized(timing, success=False, poisoned=poisoned)

    values = _pop_values(collector)
    assert values[RaidenOffloadMetricName.FAILED_LOAD_JOBS] == dict.fromkeys(blamed, 1)
    assert set(values.get(RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY, {})) == timed
    assert RaidenOffloadMetricName.LOADED_TOKENS not in values
    assert not set(values) & set(OffloadTransferMetricName)


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        pytest.param({0: _SWA, 1: _ATTN, 2: _MAMBA}, 1, id="full attention first"),
        pytest.param({0: _MAMBA, 1: _SWA}, 1, id="then a group that is not recurrent"),
        pytest.param({3: _MAMBA}, 3, id="only a recurrent group"),
        pytest.param({2: _ATTN, 1: _ATTN}, 1, id="lowest id of several"),
    ],
)
def test_select_saved_tokens_group(specs: dict[int, Any], expected: int) -> None:
    groups = [_group(group_idx, spec) for group_idx, spec in specs.items()]
    assert select_saved_tokens_group(groups).group_idx == expected


def test_compute_store_key_token_ranges() -> None:
    req_ctx = ReqContext(req_id="r1")
    attn_k0 = make_offload_key(b"a0", 0)
    attn_k1 = make_offload_key(b"a1", 0)
    tail_attn = make_offload_key(b"a_tail", 0)
    mamba_k1 = make_offload_key(b"m1", 1)
    unpositioned = make_offload_key(b"a_none", 0)
    req_ctx.set_offload_key_position(attn_k0, 16)
    req_ctx.set_offload_key_position(attn_k1, 32)
    req_ctx.set_offload_key_position(tail_attn, 40)
    req_ctx.set_offload_key_position(mamba_k1, 32)
    keys = [attn_k0, attn_k1, tail_attn, mamba_k1, unpositioned]

    # Only the counted group's keys get spans; the partial-tail key starts at
    # its chunk boundary.
    assert compute_store_key_token_ranges(
        keys, req_ctx, _group(0, _ATTN)
    ) == StoreKeyTokenRanges(
        key_ranges={attn_k0: (0, 16), attn_k1: (16, 32), tail_attn: (32, 40)},
        tokens_per_chunk=16,
        atomic=False,
    )
    assert compute_store_key_token_ranges(
        keys, req_ctx, _group(1, _MAMBA)
    ) == StoreKeyTokenRanges(
        key_ranges={mamba_k1: (16, 32)}, tokens_per_chunk=16, atomic=True
    )


@pytest.mark.parametrize(
    ("key_range", "saved", "tokens_per_chunk", "atomic", "expected"),
    [
        # A 16-token chunk has 4 sub-blocks of 4 tokens.
        pytest.param((0, 16), [True] * 4, 16, False, [(0, 16)], id="all saved"),
        pytest.param((0, 16), [False] * 4, 16, False, [], id="none saved"),
        pytest.param(
            (32, 48),
            [True, False, True, True],
            16,
            False,
            [(32, 36), (40, 48)],
            id="saved in part",
        ),
        # A partial-tail key clips its last saved sub-block to its end token.
        pytest.param(
            (32, 42), [False, False, True, True], 16, False, [(40, 42)], id="tail"
        ),
        # Recurrent state and indivisible chunks are credited whole.
        pytest.param(
            (16, 32), [False, False, False, True], 16, True, [(16, 32)], id="recurrent"
        ),
        pytest.param(
            (0, 10), [False, True, False], 10, False, [(0, 10)], id="indivisible"
        ),
    ],
)
def test_saved_sub_block_ranges(
    key_range: tuple[int, int],
    saved: list[bool],
    tokens_per_chunk: int,
    atomic: bool,
    expected: list[tuple[int, int]],
) -> None:
    assert (
        saved_sub_block_ranges(key_range, saved, tokens_per_chunk, atomic=atomic)
        == expected
    )


@pytest.mark.parametrize(
    ("ranges", "expected"),
    [
        pytest.param([(0, 16), (16, 32), (32, 40)], 40, id="adjacent"),
        pytest.param([(0, 16), (8, 24), (30, 32)], 26, id="overlapping"),
        pytest.param([], 0, id="none"),
    ],
)
def test_count_merged_tokens(ranges: list[tuple[int, int]], expected: int) -> None:
    assert count_merged_tokens(ranges) == expected


@pytest.mark.parametrize(
    ("storage_backend", "success", "saved", "latency_at", "failed"),
    [
        pytest.param(None, True, {_HOST: 64}, _HOST, {}, id="host DRAM only"),
        # With a storage tier, a save also writes host DRAM, and completes in
        # storage.
        pytest.param(
            "posix", True, {_HOST: 64, _DISK: 64}, _DISK, {}, id="storage tier"
        ),
        pytest.param(None, False, {}, None, {_HOST: 1}, id="failed, host DRAM only"),
        pytest.param("posix", False, {}, None, {_DISK: 1}, id="failed, storage tier"),
    ],
)
def test_save_records_tokens_latency_and_failures_by_destination(
    storage_backend: str | None,
    success: bool,
    saved: dict[tuple[str, str], int],
    latency_at: tuple[str, str] | None,
    failed: dict[tuple[str, str], int],
) -> None:
    clock = Mock(return_value=50.0)
    collector = _collector(storage_backend=storage_backend, clock=clock)
    timing = collector.start_save_timing(
        saved_token_ranges=[(0, 64)], num_saved_sub_hashes=4
    )
    clock.return_value += 0.005
    collector.on_save_finalized(timing, success=success)

    values = _pop_values(collector)
    assert values.get(RaidenOffloadMetricName.SAVED_TOKENS, {}) == saved
    assert values.get(RaidenOffloadMetricName.SAVE_OBSERVED_LATENCY, {}) == (
        {latency_at: [pytest.approx(0.005)]} if latency_at else {}
    )
    assert values.get(RaidenOffloadMetricName.FAILED_SAVE_JOBS, {}) == failed
    # A save that completed also counts the bytes of its 4 sub-blocks, and its
    # time. These metrics have no destination.
    assert values.get(OffloadTransferMetricName.STORE_BYTES, {}) == (
        {(): 4 * _SUB_BLOCK_BYTES} if success else {}
    )
    assert values.get(OffloadTransferMetricName.STORE_SIZE, {}) == (
        {(): [4 * _SUB_BLOCK_BYTES]} if success else {}
    )
    assert values.get(OffloadTransferMetricName.STORE_TIME, {}) == (
        {(): pytest.approx(0.005)} if success else {}
    )


def test_prom_metrics_registration_and_end_to_end_observation() -> None:
    registry = prometheus_client.CollectorRegistry()
    extra = {"cpu_bytes_to_use": 1 << 20}
    kv_transfer_config = KVTransferConfig(
        kv_connector="TPURaidenOffloadingConnector",
        kv_connector_module_path="vllm_torchtpu.offload.raiden_connector",
        kv_role="kv_both",
        kv_connector_extra_config=dict(extra),
    )
    vllm_config = cast(
        VllmConfig, SimpleNamespace(kv_transfer_config=kv_transfer_config)
    )

    prom = TPURaidenOffloadingConnector.build_prom_metrics(
        vllm_config,
        _make_prom_metric_types(registry),
        ["model_name", "engine"],
        {0: ["test-model", "0"]},
    )
    assert isinstance(prom, RaidenOffloadPromMetrics)
    assert not prom._observe_deprecated_metrics
    # The spec is named on a config copy: neither the caller's extra_config
    # nor the upstream spec registry is mutated.
    assert "spec_name" not in kv_transfer_config.kv_connector_extra_config
    assert "spec_module_path" not in kv_transfer_config.kv_connector_extra_config
    assert "TPURaidenStoreOffloadingSpec" not in OffloadingSpecFactory._registry
    # Raiden's definitions are registered, and CPUOffloadingSpec's are not.
    families = {metric.name for metric in registry.collect()}
    assert RaidenOffloadMetricName.HIT_TOKENS in families
    assert not any(name.startswith("vllm:kv_offload_cpu_") for name in families)

    clock = Mock(return_value=1.0)
    collector = _collector(storage_backend="posix", clock=clock)
    collector.record_admission_hits({CacheHitSource.DISK: 32})
    load_timing = collector.start_load_timing(
        num_host_sub_hashes=0,
        remote_sources={b"d0": CacheHitSource.DISK, b"d1": CacheHitSource.DISK},
        tokens_by_source={CacheHitSource.DISK: 32},
    )
    clock.return_value += 0.05
    load_timing.on_sub_hash_done(b"d0", clock())
    load_timing.on_sub_hash_done(b"d1", clock())
    collector.on_load_finalized(load_timing, success=True)

    save_timing = collector.start_save_timing(
        saved_token_ranges=[(0, 32)], num_saved_sub_hashes=2
    )
    clock.return_value += 0.02
    collector.on_save_finalized(save_timing, success=True)

    failed_timing = collector.start_load_timing(
        num_host_sub_hashes=0, remote_sources={b"p0": CacheHitSource.P2P}
    )
    failed_timing.on_sub_hash_failed(b"p0")
    collector.on_load_finalized(failed_timing, success=False)
    collector.record_failed_save(launched=True)

    raw_stats = collector.pop_stats()
    assert raw_stats is not None
    # The scheduler adds the manager's config info to an engine's first
    # payload. The Raiden manager reports none.
    raw_stats.set_info([{}])
    # Ship the payload through the real EngineCore -> API server codec, which
    # must preserve the label-value tuple keys.
    ipc_stats = MsgpackDecoder(SchedulerStats).decode(
        MsgpackEncoder().encode(SchedulerStats(kv_connector_stats=raw_stats.to_dict()))
    )
    assert ipc_stats.kv_connector_stats is not None
    prom.observe(ipc_stats.kv_connector_stats, engine_idx=0)

    base_labels = {"model_name": "test-model", "engine": "0"}
    # Raiden's spec declares no config info labels.
    assert registry.get_sample_value("vllm:kv_offload_config_info", base_labels) == 1.0
    disk_load_labels = {**base_labels, "source": "disk", "backend": "posix"}
    p2p_load_labels = {**base_labels, "source": "p2p", "backend": "rdma"}
    host_save_labels = {**base_labels, "destination": "host", "backend": "dram"}
    disk_save_labels = {**base_labels, "destination": "disk", "backend": "posix"}

    assert (
        registry.get_sample_value(
            "vllm:kv_offload_raiden_hit_tokens_total", disk_load_labels
        )
        == 32.0
    )
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_raiden_loaded_tokens_total", disk_load_labels
        )
        == 32.0
    )
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_raiden_load_observed_latency_seconds_count",
            disk_load_labels,
        )
        == 1.0
    )
    assert registry.get_sample_value(
        "vllm:kv_offload_raiden_load_observed_latency_seconds_sum", disk_load_labels
    ) == pytest.approx(0.05)
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_raiden_failed_load_jobs_total", p2p_load_labels
        )
        == 1.0
    )
    for save_labels in (host_save_labels, disk_save_labels):
        assert (
            registry.get_sample_value(
                "vllm:kv_offload_raiden_saved_tokens_total", save_labels
            )
            == 32.0
        )
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_raiden_save_observed_latency_seconds_count",
            disk_save_labels,
        )
        == 1.0
    )
    assert registry.get_sample_value(
        "vllm:kv_offload_raiden_save_observed_latency_seconds_sum", disk_save_labels
    ) == pytest.approx(0.02)
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_raiden_failed_save_jobs_total", disk_save_labels
        )
        == 1.0
    )
    # The load and the save moved 2 sub-blocks each.
    for sample in (
        "vllm:kv_offload_load_bytes_total",
        "vllm:kv_offload_store_bytes_total",
    ):
        assert registry.get_sample_value(sample, base_labels) == 2 * _SUB_BLOCK_BYTES
    # Deprecated CPUOffloadingSpec metric stays at its initial 0.0 value.
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_total_bytes_total",
            {**base_labels, "transfer_type": "CPU_to_GPU"},
        )
        == 0.0
    )


@pytest.mark.parametrize(
    ("tier_config", "env", "configured_tiers"),
    [
        ({}, {}, {("host", "dram")}),
        ({"raiden_storage_path": "/mnt/kv"}, {}, {("host", "dram"), ("disk", "posix")}),
        (
            {},
            {"RAIDEN_STORAGE_PATH": "/mnt/kv", "RAIDEN_STORAGE_TYPE": "POSIX"},
            {("host", "dram"), ("disk", "posix")},
        ),
        (
            {"global_registry_address": "registry:2379"},
            {},
            {("host", "dram"), ("p2p", "rdma")},
        ),
    ],
)
def test_prom_metrics_start_each_configured_tier_series_at_zero(
    monkeypatch: pytest.MonkeyPatch,
    tier_config: dict[str, Any],
    env: dict[str, str],
    configured_tiers: set[tuple[str, str]],
) -> None:
    """`rate()` misses the first increment of a series that only appears with it."""
    for name in ("RAIDEN_STORAGE_PATH", "RAIDEN_STORAGE_TYPE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    registry = prometheus_client.CollectorRegistry()
    kv_transfer_config = KVTransferConfig(
        kv_connector="TPURaidenOffloadingConnector",
        kv_connector_module_path="vllm_torchtpu.offload.raiden_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": 1 << 20, **tier_config},
    )

    TPURaidenOffloadingConnector.build_prom_metrics(
        cast(VllmConfig, SimpleNamespace(kv_transfer_config=kv_transfer_config)),
        _make_prom_metric_types(registry),
        ["model_name", "engine"],
        {0: ["test-model", "0"], 1: ["test-model", "1"]},
    )

    # Saves never go to peers, and a save completes in storage when there is a
    # storage tier.
    save_destinations = configured_tiers - {_P2P}
    save_completes_at = _DISK if _DISK in configured_tiers else _HOST
    for sample, tier_label, tiers in (
        ("vllm:kv_offload_raiden_hit_tokens_total", "source", configured_tiers),
        ("vllm:kv_offload_raiden_loaded_tokens_total", "source", configured_tiers),
        ("vllm:kv_offload_raiden_failed_load_jobs_total", "source", configured_tiers),
        (
            "vllm:kv_offload_raiden_load_observed_latency_seconds_count",
            "source",
            configured_tiers,
        ),
        ("vllm:kv_offload_raiden_saved_tokens_total", "destination", save_destinations),
        (
            "vllm:kv_offload_raiden_failed_save_jobs_total",
            "destination",
            save_destinations,
        ),
        (
            "vllm:kv_offload_raiden_save_observed_latency_seconds_count",
            "destination",
            {save_completes_at},
        ),
    ):
        for engine in ("0", "1"):
            for tier in (_HOST, _P2P, _DISK):
                value = registry.get_sample_value(
                    sample,
                    {
                        "model_name": "test-model",
                        "engine": engine,
                        tier_label: tier[0],
                        "backend": tier[1],
                    },
                )
                assert value == (0.0 if tier in tiers else None), (sample, tier)
    # vLLM's transfer metrics have no tier label: one series for each engine.
    for engine in ("0", "1"):
        for sample in (
            "vllm:kv_offload_load_bytes_total",
            "vllm:kv_offload_load_time_total",
            "vllm:kv_offload_load_size_count",
            "vllm:kv_offload_store_bytes_total",
            "vllm:kv_offload_store_time_total",
            "vllm:kv_offload_store_size_count",
        ):
            labels = {"model_name": "test-model", "engine": engine}
            assert registry.get_sample_value(sample, labels) == 0.0, sample


def test_tpu_multi_connector_builds_and_observes_raiden_offload_prom_metrics() -> None:
    registry = prometheus_client.CollectorRegistry()
    kv_transfer_config = KVTransferConfig(
        kv_connector="TPUMultiConnector",
        kv_connector_module_path=(
            "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"
        ),
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "TPURaidenConnector",
                    "kv_connector_module_path": (
                        "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
                    ),
                    "kv_role": "kv_both",
                },
                {
                    "kv_connector": "TPURaidenOffloadingConnector",
                    "kv_connector_module_path": (
                        "vllm_torchtpu.offload.raiden_connector"
                    ),
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 20},
                },
            ]
        },
    )
    vllm_config = cast(
        VllmConfig, SimpleNamespace(kv_transfer_config=kv_transfer_config)
    )
    multi_prom = TPUMultiConnector.build_prom_metrics(
        vllm_config,
        _make_prom_metric_types(registry),
        ["model_name"],
        {0: ["disagg-model"]},
    )
    assert multi_prom is not None

    collector = _collector(storage_backend="posix")
    collector.record_admission_hits({CacheHitSource.DISK: 64})
    offload_stats = collector.pop_stats()
    assert offload_stats is not None

    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorStats,
    )

    multi_stats = MultiKVConnectorStats(
        data={"TPURaidenOffloadingConnector": offload_stats}
    )
    multi_prom.observe(multi_stats.to_dict(), engine_idx=0)
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_raiden_hit_tokens_total",
            {"model_name": "disagg-model", "source": "disk", "backend": "posix"},
        )
        == 64.0
    )
