# KV Cache Offloading Metrics

With `TPURaidenOffloadingConnector`, alone or as a child of `TPUMultiConnector`, the `/metrics` endpoint shows which tier served each prefix-cache hit: the engine's host DRAM pool, the pool of another engine, or the storage tier. It also shows, for each tier, the tokens that were loaded and saved, how long the loads and saves took, and how many of them failed. For all other metrics, see [Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) in the vLLM documentation.

## `vllm:prompt_tokens_cached_by_source`

vLLM's counter of cached prompt tokens, by the tier that supplied them. The connector reports the hits that it serves as `host`, `p2p`, or `disk`:

| `source` | Tier |
| :--- | :--- |
| `device` | The prefix cache in TPU memory (HBM). |
| `host` | The engine's host DRAM pool. |
| `p2p` | The host DRAM pool of another engine. |
| `disk` | The storage tier. |
| `external_unspecified` | No tier: tokens of a hit that the connector cannot assign to a tier, for example because the blocks were evicted before the load started. |

It is a counter, so a scrape shows it as `vllm:prompt_tokens_cached_by_source_total`. It counts a request's tokens when the scheduler admits the request, and does not count them again when the request is admitted again after a preemption.

A hit can come from more than one tier:

- **Blocks in different tiers.** Each tier counts the tokens of its own blocks. For example, the first 1024 tokens of a 2048-token hit are in host DRAM and the other 1024 are in storage: `host` and `disk` each grow by 1024.
- **Two kinds of cached data for the same tokens.** A hybrid model caches attention KV and linear-attention state, and a hit needs both. Such tokens are counted once, under the slowest tier that holds a part of them: `disk` is slower than `p2p`, and `p2p` is slower than `host`. For example, the attention KV of a 2048-token hit is in host DRAM and its linear-attention state is in storage: `disk` grows by 2048, and `host` does not grow.

## `vllm:kv_offload_raiden_hit_tokens`

The connector's own counter of the prompt tokens that the offload tiers supplied. It counts the same hits as the `host`, `p2p`, and `disk` series of `vllm:prompt_tokens_cached_by_source`. Its `backend` label also names what serves the tier:

| `source` | `backend` | Tier |
| :--- | :--- | :--- |
| `host` | `dram` | The engine's host DRAM pool. |
| `p2p` | `rdma` | The host DRAM pool of another engine. |
| `disk` | `posix`, the storage type | The storage tier. |

Unlike `vllm:prompt_tokens_cached_by_source`, it has no `device` series and no `external_unspecified` series: it counts only tokens that an offload tier supplied. It is a counter, so a scrape shows it as `vllm:kv_offload_raiden_hit_tokens_total`.

### Series at startup

The `rate()` and `increase()` functions of Prometheus miss the first increment of a series that did not exist before. So the series of every configured tier are created at 0 at startup, for each engine:

```text
vllm:kv_offload_raiden_hit_tokens_total{backend="dram",engine="0",model_name="my-model",source="host"} 0.0
vllm:kv_offload_raiden_hit_tokens_total{backend="posix",engine="0",model_name="my-model",source="disk"} 0.0
```

| Tier | Configured when |
| :--- | :--- |
| `host` | Always. |
| `p2p` | `global_registry_address` is set in `kv_connector_extra_config`. |
| `disk` | A storage path is set, through `raiden_storage_path` in `kv_connector_extra_config` or through the `RAIDEN_STORAGE_PATH` environment variable. |

> [!NOTE]
> The API server process creates these series from its own settings: `kv_connector_extra_config`, and the `RAIDEN_STORAGE_PATH` and `RAIDEN_STORAGE_TYPE` environment variables. A single `vllm serve` command gives every process the same settings. In [multi-host serving](../user_guide/multihost.md), each host runs its own `vllm serve`, so set the storage tier on the host that serves `/metrics` as on the other hosts. Otherwise a `disk` series appears only with its first hit from storage.

## Load metrics

After a hit, the connector loads the cached blocks from their tiers into TPU memory (HBM). Three metrics follow these loads. They have the `source` and `backend` labels of `vllm:kv_offload_raiden_hit_tokens`, and their series are created at 0 at startup in the same way.

| Metric | Type | Value |
| :--- | :--- | :--- |
| `vllm:kv_offload_raiden_loaded_tokens` | Counter | The tokens that were loaded from the tier. |
| `vllm:kv_offload_raiden_load_observed_latency_seconds` | Histogram | The time that the tier's part of a load took, in seconds. |
| `vllm:kv_offload_raiden_failed_load_jobs` | Counter | The loads in which the tier failed. |

**Loaded tokens.** A load counts its tokens when all of it has completed, with the split by tier of its hit. It differs from `vllm:kv_offload_raiden_hit_tokens` in two cases:

- A load that fails counts no tokens, because vLLM discards its blocks and computes them again.
- A request that is admitted again after a preemption loads its blocks again. The second load counts its tokens. The hit is not counted again.

**Latency.** A load can read from several tiers at the same time, and each tier gets its own sample: the time from the start of the load until the scheduler sees that the last block from this tier has arrived. So in one load, a slow read from storage does not hide a fast read from host DRAM. The scheduler looks for completed transfers once per step, so a sample is rounded up to the next step. While a batch runs, a step takes at least one forward pass.

**Failed loads.** A failed load counts once for each tier that failed: its read was refused, its transfer failed, or it was still running when the scheduler stopped waiting for the load. A tier that completed its part is not counted. For example, a load reads from host DRAM and from storage, and the read from storage fails: `disk` grows by 1 and `host` does not, and the load still records the latency of `host`. A load that the scheduler stopped waiting for records no latency.

A hit whose blocks were evicted before the load started appears in none of the three metrics, because no transfer started.

## Save metrics

The connector saves the KV cache of a request from TPU memory (HBM) to the offload tiers. Three metrics follow these saves. In place of `source`, they have a `destination` label, for the tier that the save writes:

| Metric | Type | Value |
| :--- | :--- | :--- |
| `vllm:kv_offload_raiden_saved_tokens` | Counter | The tokens that were saved to the tier. |
| `vllm:kv_offload_raiden_save_observed_latency_seconds` | Histogram | The time that a save took, in seconds. |
| `vllm:kv_offload_raiden_failed_save_jobs` | Counter | The saves that failed. |

| `destination` | `backend` | Tier |
| :--- | :--- | :--- |
| `host` | `dram` | The engine's host DRAM pool. Every save writes it. |
| `disk` | `posix`, the storage type | The storage tier. With a storage tier, a save writes host DRAM and storage, and is complete when storage has the data. |

There is no `p2p` destination: an engine does not save into the pool of another engine.

**Saved tokens.** A save counts its tokens when it has completed. With a storage tier, `host` and `disk` both grow by the tokens of the save. A save counts only the tokens that it writes, and each of them once:

- A hybrid model saves attention KV and linear-attention state for the same tokens. Only the attention KV counts.
- A block that is already in host DRAM is not saved again, and counts no tokens.
- A save that fails counts no tokens.

**Latency.** Each save that completed records one sample, under the tier where saves complete: `disk` with a storage tier, `host` without. The sample is the time from the start of the save until the scheduler sees it complete. It includes retries, and it is rounded up to the next scheduler step, like the load latency.

**Failed saves.** A save that could not start counts under `host`, because the host DRAM pool refused it. A save that started and then failed counts under the tier where saves complete.

The series of these metrics are created at 0 at startup too: the two counters for `host` and for a configured storage tier, and the latency for the tier where saves complete.

## Metrics that are not listed

vLLM's CPU offloading backend reports on the host memory pool that it manages: `vllm:kv_offload_cpu_cache_usage_perc`, `vllm:kv_offload_cpu_cache_read_usage_perc`, `vllm:kv_offload_cpu_cache_write_usage_perc`, `vllm:kv_offload_cpu_allocation_size`, and the `cpu_*` labels of `vllm:kv_offload_config_info`. This connector keeps offloaded blocks in the Raiden store, which manages the host DRAM pool itself. vLLM's pool does not exist here, so nothing sets these metrics, and `/metrics` does not list them.
