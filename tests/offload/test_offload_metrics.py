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

import enum
from types import SimpleNamespace
from typing import Any, cast

import prometheus_client
import pytest
import torch
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import PromMetric
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    get_connector_metric_definitions,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector
from vllm_torchtpu.offload.metrics import (
    HOST_OFFLOAD_LABELS,
    OFFLOAD_LOAD_LABELNAMES,
    OFFLOAD_SAVE_LABELNAMES,
    P2P_OFFLOAD_LABELS,
    UNSPECIFIED_OFFLOAD_LABELS,
    LoadTimingState,
    OffloadTransferMetricName,
    RaidenOffloadMetricName,
    RaidenOffloadPromMetrics,
    RaidenOffloadStatsCollector,
    build_raiden_offload_metric_definitions,
    compute_store_key_token_ranges,
    count_merged_tokens,
    external_sources_with_tokens,
    saved_sub_block_ranges,
    select_saved_tokens_group,
    split_external_hit,
    status_to_hit_source,
)
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector


class _FakeBlockStatus(enum.Enum):
    INIT = 0
    REMOTE = 1
    HOST = 2
    HBM = 3
    HOST_AND_HBM = 4
    SHARED_STORAGE = 5
    CXL = 6


class _FakeClock:
    def __init__(self, initial: float = 100.0) -> None:
        self.t = initial

    def __call__(self) -> float:
        return self.t

    def advance(self, delta: float) -> None:
        self.t += delta


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


def test_metric_definitions_have_directional_labels() -> None:
    assert UNSPECIFIED_OFFLOAD_LABELS == ("external_unspecified", "unknown")
    defs = build_raiden_offload_metric_definitions({})
    assert set(defs.keys()) == {m.value for m in RaidenOffloadMetricName}
    load_metrics = {
        RaidenOffloadMetricName.HIT_TOKENS,
        RaidenOffloadMetricName.LOADED_TOKENS,
        RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY,
        RaidenOffloadMetricName.FAILED_LOAD_JOBS,
    }
    save_metrics = {
        RaidenOffloadMetricName.SAVED_TOKENS,
        RaidenOffloadMetricName.SAVE_OBSERVED_LATENCY,
        RaidenOffloadMetricName.FAILED_SAVE_JOBS,
    }
    for metric in load_metrics:
        assert defs[metric].labelnames == OFFLOAD_LOAD_LABELNAMES
    for metric in save_metrics:
        assert defs[metric].labelnames == OFFLOAD_SAVE_LABELNAMES


@pytest.mark.parametrize(
    ("status", "expected_hit"),
    [
        (_FakeBlockStatus.HOST, CacheHitSource.HOST),
        (_FakeBlockStatus.HOST_AND_HBM, CacheHitSource.HOST),
        (_FakeBlockStatus.REMOTE, CacheHitSource.P2P),
        (_FakeBlockStatus.SHARED_STORAGE, CacheHitSource.DISK),
        (_FakeBlockStatus.HBM, CacheHitSource.EXTERNAL_UNSPECIFIED),
        (_FakeBlockStatus.INIT, CacheHitSource.EXTERNAL_UNSPECIFIED),
        (_FakeBlockStatus.CXL, CacheHitSource.EXTERNAL_UNSPECIFIED),
    ],
)
def test_status_to_hit_source(
    status: _FakeBlockStatus,
    expected_hit: CacheHitSource,
) -> None:
    assert status_to_hit_source(status, _FakeBlockStatus) == expected_hit


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


def test_collector_unknown_sources_and_missing_key_handling() -> None:
    clock = _FakeClock(5.0)
    collector = RaidenOffloadStatsCollector(
        bytes_per_sub_block=256,
        storage_backend=None,
        clock=clock,
    )

    # 1. Disk source without configured storage_backend uses ("disk", "unknown"),
    # EXTERNAL_UNSPECIFIED uses ("external_unspecified", "unknown"), and DEVICE
    # or non-CacheHitSource values raise.
    assert collector.source_labels(CacheHitSource.DISK) == ("disk", "unknown")
    assert (
        collector.source_labels(CacheHitSource.EXTERNAL_UNSPECIFIED)
        == UNSPECIFIED_OFFLOAD_LABELS
    )
    for invalid in (CacheHitSource.DEVICE, "host"):
        with pytest.raises(ValueError, match="Unsupported offload source"):
            collector.source_labels(cast(Any, invalid))

    # 2. Missing key in _step_key_sources returns None from get_load_sub_sources
    # and is skipped by compute_load_tokens_by_source rather than mislabeled.
    missing_key = make_offload_key(b"missing", 0)
    assert collector.get_load_sub_sources(missing_key) is None
    assert (
        collector.compute_load_tokens_by_source(
            start_token=0,
            end_token=16,
            load_key_ranges=[(0, 16, missing_key)],
            tokens_per_chunk_by_group={0: 16},
        )
        == {}
    )
    assert collector.pop_stats() is None

    # 3. Unrecognized BlockStatus (CXL) in prepare_load / start_load_timing
    # records under ("external_unspecified", "unknown") instead of ("p2p", "rdma").
    k0 = make_offload_key(b"k0", 0)
    cxl_block = SimpleNamespace(status=_FakeBlockStatus.CXL)
    collector.record_prepared_load(
        [k0],
        1,
        {b"s0": cxl_block},
        [b"s0"],
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k0) == (
        (CacheHitSource.EXTERNAL_UNSPECIFIED, 1),
    )

    collector.record_prepared_load(
        [k0],
        2,
        {b"s1": cxl_block},
        [b"s0", b"s1"],
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k0) == (
        (CacheHitSource.HOST, 1),
        (CacheHitSource.EXTERNAL_UNSPECIFIED, 1),
    )

    tokens_by_source = collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=16,
        load_key_ranges=[(0, 16, k0)],
        tokens_per_chunk_by_group={0: 16},
    )
    assert tokens_by_source == {
        CacheHitSource.HOST: 8,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 8,
    }
    collector.record_admission_hits(tokens_by_source)
    timing = collector.start_load_timing(
        local_hashes=[b"s0"],
        remote_hashes=[b"s1"],
        remote_slices=[cxl_block],
        block_status_cls=_FakeBlockStatus,
        tokens_by_source=tokens_by_source,
    )
    clock.advance(0.002)
    timing.on_sub_hash_done(b"s0", clock())
    clock.advance(0.007)
    timing.on_sub_hash_done(b"s1", clock())
    collector.on_load_finalized(timing, success=True, now=clock())

    stats = collector.pop_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[RaidenOffloadMetricName.HIT_TOKENS] == {
        HOST_OFFLOAD_LABELS: 8,
        UNSPECIFIED_OFFLOAD_LABELS: 8,
    }
    assert values[RaidenOffloadMetricName.LOADED_TOKENS] == {
        HOST_OFFLOAD_LABELS: 8,
        UNSPECIFIED_OFFLOAD_LABELS: 8,
    }
    assert (
        P2P_OFFLOAD_LABELS not in values[RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY]
    )
    assert values[RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY] == {
        HOST_OFFLOAD_LABELS: [pytest.approx(0.002)],
        UNSPECIFIED_OFFLOAD_LABELS: [pytest.approx(0.009)],
    }

    # 4. Failed transfer on an unrecognized BlockStatus records under
    # ("external_unspecified", "unknown").
    fail_timing = collector.start_load_timing(
        local_hashes=[],
        remote_hashes=[b"s2"],
        remote_slices=[cxl_block],
        block_status_cls=_FakeBlockStatus,
        tokens_by_source={CacheHitSource.EXTERNAL_UNSPECIFIED: 16},
    )
    fail_timing.on_sub_hash_failed(b"s2")
    collector.on_load_finalized(fail_timing, success=False, now=clock())

    fail_stats = collector.pop_stats()
    assert fail_stats is not None
    fail_values = fail_stats.data["data"]
    assert fail_values[RaidenOffloadMetricName.FAILED_LOAD_JOBS] == {
        UNSPECIFIED_OFFLOAD_LABELS: 1,
    }
    assert (
        HOST_OFFLOAD_LABELS not in fail_values[RaidenOffloadMetricName.FAILED_LOAD_JOBS]
    )
    assert (
        P2P_OFFLOAD_LABELS not in fail_values[RaidenOffloadMetricName.FAILED_LOAD_JOBS]
    )


def test_compute_load_tokens_by_source_sub_block_and_mamba() -> None:
    collector = RaidenOffloadStatsCollector(storage_backend="posix")
    peer_blk = SimpleNamespace(status=_FakeBlockStatus.REMOTE)
    disk_blk = SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE)

    # 1. Single attention key (64 tokens, 4 x 16-token sub-blocks):
    # 3 HOST + 1 DISK sub-blocks form 2 runs -> 48 HOST + 16 DISK.
    k_dense = make_offload_key(b"dense0", 0)
    collector.record_prepared_load(
        [k_dense],
        4,
        {b"s3": disk_blk},
        [b"s0", b"s1", b"s2", b"s3"],
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k_dense) == (
        (CacheHitSource.HOST, 3),
        (CacheHitSource.DISK, 1),
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=64,
        load_key_ranges=[(0, 64, k_dense)],
        tokens_per_chunk_by_group={0: 64},
    ) == {
        CacheHitSource.HOST: 48,
        CacheHitSource.DISK: 16,
    }

    # 2. Partial-tail range [0, 48) on a 64-token chunk excludes t3 ([48, 64)).
    k_tail = make_offload_key(b"tail0", 0)
    collector.record_prepared_load(
        [k_tail],
        4,
        {b"t2": peer_blk, b"t3": disk_blk},
        [b"t0", b"t1", b"t2", b"t3"],
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k_tail) == (
        (CacheHitSource.HOST, 2),
        (CacheHitSource.P2P, 1),
        (CacheHitSource.DISK, 1),
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=48,
        load_key_ranges=[(0, 48, k_tail)],
        tokens_per_chunk_by_group={0: 64},
    ) == {
        CacheHitSource.HOST: 32,
        CacheHitSource.P2P: 16,
    }

    # 3. Large sub-block factor (factor=512, 8192-token chunks):
    # - Uniform HOST batch shares a single 1-element tuple across keys.
    # - Contiguous 384 HOST + 128 DISK collapses to 2 runs instead of 512.
    k_big0 = make_offload_key(b"big0", 0)
    k_big1 = make_offload_key(b"big1", 0)
    collector.record_prepared_load([k_big0, k_big1], 512, {}, [], _FakeBlockStatus)
    runs_big0 = collector.get_load_sub_sources(k_big0)
    runs_big1 = collector.get_load_sub_sources(k_big1)
    assert runs_big0 == ((CacheHitSource.HOST, 512),)
    assert runs_big0 is runs_big1

    k_big_mixed = make_offload_key(b"big_mixed", 0)
    big_hashes = [f"bh_{i}".encode() for i in range(512)]
    collector.record_prepared_load(
        [k_big_mixed],
        512,
        {h: disk_blk for h in big_hashes[384:]},
        big_hashes,
        _FakeBlockStatus,
    )
    assert collector.get_load_sub_sources(k_big_mixed) == (
        (CacheHitSource.HOST, 384),
        (CacheHitSource.DISK, 128),
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=16384,
        load_key_ranges=[(0, 8192, k_big0), (8192, 16384, k_big_mixed)],
        tokens_per_chunk_by_group={0: 8192},
    ) == {
        CacheHitSource.HOST: 8192 + 6144,
        CacheHitSource.DISK: 2048,
    }

    # 4. Hybrid Attention (group 0) + Mamba (group 1) over [0, 32):
    # k_a0=(HOST*4), k_a1=(HOST*2, P2P*2), k_m1=(HOST*4) -> 24 HOST + 8 P2P.
    k_a0 = make_offload_key(b"a0", 0)
    k_a1 = make_offload_key(b"a1", 0)
    k_m1 = make_offload_key(b"m1", 1)
    collector.record_prepared_load([k_a0, k_m1], 4, {}, [], _FakeBlockStatus)
    collector.record_prepared_load(
        [k_a1],
        4,
        {b"a1_2": peer_blk, b"a1_3": peer_blk},
        [b"a1_0", b"a1_1", b"a1_2", b"a1_3"],
        _FakeBlockStatus,
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=32,
        load_key_ranges=[(0, 16, k_a0), (16, 32, k_a1), (0, 32, k_m1)],
        tokens_per_chunk_by_group={0: 16, 1: 16},
        mamba_group_ids={1},
    ) == {
        CacheHitSource.HOST: 24,
        CacheHitSource.P2P: 8,
    }

    # If Mamba's indivisible checkpoint has one byte slice in DISK, the whole
    # prefix [0, 32) depends on that DISK checkpoint -> {DISK: 32}.
    collector.record_prepared_load(
        [k_m1],
        4,
        {b"m1_3": disk_blk},
        [b"m1_0", b"m1_1", b"m1_2", b"m1_3"],
        _FakeBlockStatus,
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=32,
        load_key_ranges=[(0, 16, k_a0), (16, 32, k_a1), (0, 32, k_m1)],
        tokens_per_chunk_by_group={0: 16, 1: 16},
        mamba_group_ids={1},
    ) == {
        CacheHitSource.DISK: 32,
    }


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
    collector = RaidenOffloadStatsCollector(storage_backend="posix")
    disk_blk = SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE)
    # A 64-token key with four 16-token sub-blocks: two in host DRAM, two on disk.
    key = make_offload_key(b"mixed", 0)
    collector.record_prepared_load(
        [key],
        4,
        {b"s2": disk_blk, b"s3": disk_blk},
        [b"s0", b"s1", b"s2", b"s3"],
        _FakeBlockStatus,
    )
    assert (
        collector.compute_load_tokens_by_source(
            start_token=start_token,
            end_token=end_token,
            load_key_ranges=[(*key_range, key)],
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
    collector = RaidenOffloadStatsCollector(storage_backend="posix")
    disk_blk = SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE)
    key = make_offload_key(b"mixed", 0)
    collector.record_prepared_load(
        [key],
        4,
        {b"s3": disk_blk},
        [b"s0", b"s1", b"s2", b"s3"],
        _FakeBlockStatus,
    )
    assert collector.compute_load_tokens_by_source(
        start_token=0,
        end_token=64,
        load_key_ranges=[(0, 64, key)],
        tokens_per_chunk_by_group=tokens_per_chunk_by_group,
        mamba_group_ids=mamba_group_ids,
    ) == {CacheHitSource.DISK: 64}
    assert len(warnings) == num_warnings


def test_compute_store_key_token_ranges_and_merged_tokens() -> None:
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

    # Only the counted group's keys get spans; the partial-tail key starts at
    # its chunk boundary.
    ranges = compute_store_key_token_ranges(
        [attn_k0, attn_k1, tail_attn, mamba_k1, unpositioned],
        req_ctx,
        group_idx=0,
        tokens_per_chunk=16,
    )
    assert ranges == {
        attn_k0: (0, 16),
        attn_k1: (16, 32),
        tail_attn: (32, 40),
    }
    assert count_merged_tokens(ranges.values()) == 40
    assert count_merged_tokens([(0, 16), (8, 24), (30, 32)]) == 26
    assert count_merged_tokens([]) == 0


def test_saved_sub_block_ranges() -> None:
    # 16-token chunk, 4 sub-blocks of 4 tokens each.
    assert saved_sub_block_ranges((0, 16), [True] * 4, 16, atomic=False) == [(0, 16)]
    assert saved_sub_block_ranges((0, 16), [False] * 4, 16, atomic=False) == []
    assert saved_sub_block_ranges(
        (32, 48), [True, False, True, True], 16, atomic=False
    ) == [(32, 36), (40, 48)]
    # A partial-tail key clips its last saved sub-block to its end token.
    assert saved_sub_block_ranges(
        (32, 42), [False, False, True, True], 16, atomic=False
    ) == [(40, 42)]
    # Recurrent state and indivisible chunks are credited whole.
    assert saved_sub_block_ranges(
        (16, 32), [False, False, False, True], 16, atomic=True
    ) == [(16, 32)]
    assert saved_sub_block_ranges((0, 10), [False, True, False], 10, atomic=False) == [
        (0, 10)
    ]


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


def test_select_saved_tokens_group() -> None:
    attn = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    mamba = MambaSpec(
        block_size=16,
        shapes=((2, 8),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
    )
    assert select_saved_tokens_group({0: swa, 1: attn, 2: mamba}) == 1
    assert select_saved_tokens_group({0: mamba, 1: swa}) == 1
    assert select_saved_tokens_group({3: mamba}) == 3
    assert select_saved_tokens_group({}) is None


def test_collector_load_lifecycle_staggered_sources_and_partial_failure() -> None:
    clock = _FakeClock(10.0)
    collector = RaidenOffloadStatsCollector(
        bytes_per_sub_block=1024,
        storage_backend="posix",
        clock=clock,
    )

    # Preempted admission does not increment hit_tokens; initial admission does.
    tokens = {
        CacheHitSource.HOST: 16,
        CacheHitSource.P2P: 32,
        CacheHitSource.DISK: 48,
    }
    collector.record_admission_hits(tokens, preempted=True)
    assert collector.pop_stats() is None

    collector.record_admission_hits(tokens, preempted=False)
    timing = collector.start_load_timing(
        local_hashes=[b"h0", b"h1"],
        remote_hashes=[b"p0", b"d0"],
        remote_slices=[
            SimpleNamespace(status=_FakeBlockStatus.REMOTE),
            SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE),
        ],
        block_status_cls=_FakeBlockStatus,
        tokens_by_source=tokens,
    )
    clock.advance(0.002)
    timing.on_sub_hash_done(b"h0", clock())
    timing.on_sub_hash_done(b"h1", clock())
    clock.advance(0.006)
    timing.on_sub_hash_done(b"p0", clock())
    clock.advance(0.032)
    timing.on_sub_hash_done(b"d0", clock())
    collector.on_load_finalized(timing, success=True, now=clock())

    stats = collector.pop_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[RaidenOffloadMetricName.HIT_TOKENS] == {
        HOST_OFFLOAD_LABELS: 16,
        P2P_OFFLOAD_LABELS: 32,
        ("disk", "posix"): 48,
    }
    assert values[RaidenOffloadMetricName.LOADED_TOKENS] == {
        HOST_OFFLOAD_LABELS: 16,
        P2P_OFFLOAD_LABELS: 32,
        ("disk", "posix"): 48,
    }
    assert values[OffloadTransferMetricName.LOAD_BYTES] == {(): 4 * 1024}
    assert values[OffloadTransferMetricName.LOAD_TIME][()] == pytest.approx(0.04)
    assert values[OffloadTransferMetricName.LOAD_SIZE] == {(): [4 * 1024]}
    assert values[RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY][
        HOST_OFFLOAD_LABELS
    ] == [pytest.approx(0.002)]
    assert values[RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY][
        P2P_OFFLOAD_LABELS
    ] == [pytest.approx(0.008)]
    assert values[RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY][("disk", "posix")] == [
        pytest.approx(0.04)
    ]

    # Partial failure: host finishes cleanly, disk fails -> host latency kept,
    # zero loaded_tokens, and failed_load_jobs incremented only for ("disk", "posix").
    fail_timing = collector.start_load_timing(
        local_hashes=[b"h2"],
        remote_hashes=[b"d2"],
        remote_slices=[SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE)],
        block_status_cls=_FakeBlockStatus,
        tokens_by_source={CacheHitSource.HOST: 16, CacheHitSource.DISK: 16},
    )
    clock.advance(0.003)
    fail_timing.on_sub_hash_done(b"h2", clock())
    clock.advance(0.010)
    fail_timing.on_sub_hash_failed(b"d2")
    collector.on_load_finalized(fail_timing, success=False, now=clock())

    fail_stats = collector.pop_stats()
    assert fail_stats is not None
    fail_values = fail_stats.data["data"]
    assert RaidenOffloadMetricName.LOADED_TOKENS not in fail_values
    assert fail_values[RaidenOffloadMetricName.FAILED_LOAD_JOBS] == {
        ("disk", "posix"): 1
    }
    assert fail_values[RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY] == {
        HOST_OFFLOAD_LABELS: [pytest.approx(0.003)]
    }


def test_load_timing_blames_only_failed_or_in_flight_tiers() -> None:
    collector = RaidenOffloadStatsCollector(storage_backend="posix")

    def start() -> LoadTimingState:
        return collector.start_load_timing(
            local_hashes=[b"h0"],
            remote_hashes=[b"d0"],
            remote_slices=[SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE)],
            block_status_cls=_FakeBlockStatus,
        )

    def finalize_failed(
        timing: LoadTimingState, *, poisoned: bool = False
    ) -> dict[str, Any]:
        collector.on_load_finalized(timing, success=False, poisoned=poisoned)
        stats = collector.pop_stats()
        assert stats is not None
        return stats.data["data"]

    # store.load() refused: read_remote() never launched, so disk is not blamed.
    timing = start()
    timing.mark_local_launch_failed()
    assert finalize_failed(timing)[RaidenOffloadMetricName.FAILED_LOAD_JOBS] == {
        HOST_OFFLOAD_LABELS: 1
    }

    timing = start()
    timing.mark_remote_launch_failed([b"d0"])
    assert finalize_failed(timing)[RaidenOffloadMetricName.FAILED_LOAD_JOBS] == {
        ("disk", "posix"): 1
    }

    # A later read_remote() batch refused while the first (one peer plus the
    # storage reads) is in flight: only the never-launched batch's tier is
    # blamed, and the storage tier still records its latency.
    timing = collector.start_load_timing(
        local_hashes=[],
        remote_hashes=[b"p0", b"d0", b"p1"],
        remote_slices=[
            SimpleNamespace(status=_FakeBlockStatus.REMOTE),
            SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE),
            SimpleNamespace(status=_FakeBlockStatus.REMOTE),
        ],
        block_status_cls=_FakeBlockStatus,
    )
    timing.mark_remote_launch_failed([b"p1"])
    timing.on_sub_hash_done(b"p0", 1.0)
    timing.on_sub_hash_done(b"d0", 1.0)
    values = finalize_failed(timing)
    assert values[RaidenOffloadMetricName.FAILED_LOAD_JOBS] == {("p2p", "rdma"): 1}
    assert set(values[RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY]) == {
        ("disk", "posix")
    }

    # Poisoned after the host tier landed: only the in-flight disk tier is
    # blamed, and a poisoned job records no latency.
    timing = start()
    timing.on_sub_hash_done(b"h0", 1.0)
    timing.mark_poisoned()
    timing.on_sub_hash_done(b"d0", 2.0)
    values = finalize_failed(timing, poisoned=True)
    assert values[RaidenOffloadMetricName.FAILED_LOAD_JOBS] == {("disk", "posix"): 1}
    assert RaidenOffloadMetricName.LOAD_OBSERVED_LATENCY not in values


def test_collector_save_lifecycle_dram_only_vs_storage() -> None:
    clock = _FakeClock(50.0)
    k0 = make_offload_key(b"k0", 0)
    ranges = {k0: (0, 64)}

    # DRAM-only save credits ("host", "dram") only.
    dram_collector = RaidenOffloadStatsCollector(
        bytes_per_sub_block=512,
        storage_backend=None,
        clock=clock,
    )
    save_timing = dram_collector.start_save_timing(
        saved_token_ranges=ranges.values(),
        num_saved_sub_hashes=4,
    )
    clock.advance(0.005)
    dram_collector.on_save_finalized(save_timing, success=True, now=clock())
    dram_stats = dram_collector.pop_stats()
    assert dram_stats is not None
    dram_values = dram_stats.data["data"]
    assert dram_values[RaidenOffloadMetricName.SAVED_TOKENS] == {
        HOST_OFFLOAD_LABELS: 64
    }
    assert dram_values[RaidenOffloadMetricName.SAVE_OBSERVED_LATENCY] == {
        HOST_OFFLOAD_LABELS: [pytest.approx(0.005)]
    }
    assert dram_values[OffloadTransferMetricName.STORE_BYTES] == {(): 2048}

    # Write-through storage save credits saved_tokens to both ("host", "dram")
    # and ("disk", "posix"), and records save_observed_latency under ("disk", "posix").
    storage_collector = RaidenOffloadStatsCollector(
        bytes_per_sub_block=512,
        storage_backend="posix",
        clock=clock,
    )
    wt_timing = storage_collector.start_save_timing(
        saved_token_ranges=ranges.values(),
        num_saved_sub_hashes=4,
    )
    clock.advance(0.025)
    storage_collector.on_save_finalized(wt_timing, success=True, now=clock())
    wt_stats = storage_collector.pop_stats()
    assert wt_stats is not None
    wt_values = wt_stats.data["data"]
    assert wt_values[RaidenOffloadMetricName.SAVED_TOKENS] == {
        HOST_OFFLOAD_LABELS: 64,
        ("disk", "posix"): 64,
    }
    assert wt_values[RaidenOffloadMetricName.SAVE_OBSERVED_LATENCY] == {
        ("disk", "posix"): [pytest.approx(0.025)]
    }

    # Failed save records failed_save_jobs on destination and 0 saved_tokens.
    fail_timing = storage_collector.start_save_timing(
        saved_token_ranges=ranges.values(),
        num_saved_sub_hashes=4,
    )
    storage_collector.on_save_finalized(fail_timing, success=False, now=clock())
    fail_stats = storage_collector.pop_stats()
    assert fail_stats is not None
    fail_values = fail_stats.data["data"]
    assert RaidenOffloadMetricName.SAVED_TOKENS not in fail_values
    assert fail_values[RaidenOffloadMetricName.FAILED_SAVE_JOBS] == {
        ("disk", "posix"): 1
    }


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

    clock = _FakeClock(1.0)
    collector = RaidenOffloadStatsCollector(
        bytes_per_sub_block=256,
        storage_backend="posix",
        clock=clock,
    )
    collector.record_admission_hits({CacheHitSource.DISK: 32})
    load_timing = collector.start_load_timing(
        local_hashes=[],
        remote_hashes=[b"s0", b"s1"],
        remote_slices=[
            SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE),
            SimpleNamespace(status=_FakeBlockStatus.SHARED_STORAGE),
        ],
        block_status_cls=_FakeBlockStatus,
        tokens_by_source={CacheHitSource.DISK: 32},
    )
    clock.advance(0.05)
    load_timing.on_sub_hash_done(b"s0", clock())
    load_timing.on_sub_hash_done(b"s1", clock())
    collector.on_load_finalized(load_timing, success=True, now=clock())

    save_timing = collector.start_save_timing(
        saved_token_ranges=[(0, 32)],
        num_saved_sub_hashes=2,
    )
    clock.advance(0.02)
    collector.on_save_finalized(save_timing, success=True, now=clock())

    p2p_fail_timing = collector.start_load_timing(
        local_hashes=[],
        remote_hashes=[b"p_fail"],
        remote_slices=[SimpleNamespace(status=_FakeBlockStatus.REMOTE)],
        block_status_cls=_FakeBlockStatus,
        tokens_by_source={CacheHitSource.P2P: 16},
    )
    p2p_fail_timing.on_sub_hash_failed(b"p_fail")
    collector.on_load_finalized(p2p_fail_timing, success=False, now=clock())
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
        registry.get_sample_value("vllm:kv_offload_hit_tokens_total", disk_load_labels)
        == 32.0
    )
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_loaded_tokens_total", disk_load_labels
        )
        == 32.0
    )
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_saved_tokens_total", host_save_labels
        )
        == 32.0
    )
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_saved_tokens_total", disk_save_labels
        )
        == 32.0
    )
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_load_observed_latency_seconds_count", disk_load_labels
        )
        == 1.0
    )
    assert registry.get_sample_value(
        "vllm:kv_offload_load_observed_latency_seconds_sum", disk_load_labels
    ) == pytest.approx(0.05)
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_save_observed_latency_seconds_count", disk_save_labels
        )
        == 1.0
    )
    assert registry.get_sample_value(
        "vllm:kv_offload_save_observed_latency_seconds_sum", disk_save_labels
    ) == pytest.approx(0.02)
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_failed_load_jobs_total", p2p_load_labels
        )
        == 1.0
    )
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_failed_save_jobs_total", disk_save_labels
        )
        == 1.0
    )
    assert (
        registry.get_sample_value("vllm:kv_offload_load_bytes_total", base_labels)
        == 512.0
    )
    assert (
        registry.get_sample_value("vllm:kv_offload_store_bytes_total", base_labels)
        == 512.0
    )
    # Deprecated CPUOffloadingSpec metric stays at its initial 0.0 value.
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_total_bytes_total",
            {**base_labels, "transfer_type": "CPU_to_GPU"},
        )
        == 0.0
    )


@pytest.mark.parametrize(
    ("tier_config", "configured_tiers", "save_destinations", "save_completes_at"),
    [
        ({}, {("host", "dram")}, {("host", "dram")}, ("host", "dram")),
        (
            {"raiden_storage_path": "/mnt/kv"},
            {("host", "dram"), ("disk", "posix")},
            {("host", "dram"), ("disk", "posix")},
            ("disk", "posix"),
        ),
        (
            {"global_registry_address": "registry:2379"},
            {("host", "dram"), ("p2p", "rdma")},
            {("host", "dram")},
            ("host", "dram"),
        ),
    ],
)
def test_prom_metrics_start_series_at_zero(
    monkeypatch: pytest.MonkeyPatch,
    tier_config: dict[str, Any],
    configured_tiers: set[tuple[str, str]],
    save_destinations: set[tuple[str, str]],
    save_completes_at: tuple[str, str],
) -> None:
    """`rate()` misses the first increment of a series that only appears with it."""
    for name in ("RAIDEN_STORAGE_PATH", "RAIDEN_STORAGE_TYPE"):
        monkeypatch.delenv(name, raising=False)
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

    for sample, tier_label, tiers in (
        ("vllm:kv_offload_hit_tokens_total", "source", configured_tiers),
        ("vllm:kv_offload_loaded_tokens_total", "source", configured_tiers),
        ("vllm:kv_offload_failed_load_jobs_total", "source", configured_tiers),
        (
            "vllm:kv_offload_load_observed_latency_seconds_count",
            "source",
            configured_tiers,
        ),
        ("vllm:kv_offload_saved_tokens_total", "destination", save_destinations),
        ("vllm:kv_offload_failed_save_jobs_total", "destination", save_destinations),
        (
            "vllm:kv_offload_save_observed_latency_seconds_count",
            "destination",
            {save_completes_at},
        ),
    ):
        for engine in ("0", "1"):
            for tier in (("host", "dram"), ("p2p", "rdma"), ("disk", "posix")):
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

    collector = RaidenOffloadStatsCollector(storage_backend="posix")
    collector.record_admission_hits({CacheHitSource.DISK: 64})
    collector.record_failed_save(launched=True)
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
            "vllm:kv_offload_hit_tokens_total",
            {"model_name": "disagg-model", "source": "disk", "backend": "posix"},
        )
        == 64.0
    )
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_failed_save_jobs_total",
            {"model_name": "disagg-model", "destination": "disk", "backend": "posix"},
        )
        == 1.0
    )
