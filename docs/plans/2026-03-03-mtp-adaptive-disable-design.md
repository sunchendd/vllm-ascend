# MTP 投机解码自适应启停设计文档

**日期**: 2026-03-03  
**状态**: 待实现  
**目标**: 在不同并发场景下，自适应启停 MTP (Multi-Token Prediction) 投机解码，以在低并发时利用投机推理的吞吐优势，在高并发时避免其额外开销。

---

## 问题背景

### 假设

MTP 投机解码在 DeepSeek-V3/R1 等模型上，存在以下性能特征：

| 场景 | NPU 利用率 | MTP 效果 | 原因 |
|---|---|---|---|
| **低并发**（少量请求） | 低（算力闲置） | **好** | Draft 模型填补闲置算力；k+1 tokens/step，吞吐大幅提升 |
| **高并发**（大量请求） | 高（算力饱和） | **差** | Draft 模型额外 forward 抢占主模型算力；批越大接受率越低，无效推测增多 |

### 关键指标

- **优化目标**：整体 Throughput（output tokens/sec）
- **触发信号**：`num_running_requests`（当前并发请求数）

---

## 调研验证方案

在实现自适应优化之前，需通过 benchmark 验证上述假设。

### Benchmark 脚本

位置：`benchmarks/benchmark_mtp_concurrency.py`

**测试矩阵**：

| 维度 | 取值 |
|---|---|
| 并发数 | `[1, 2, 4, 8, 16, 32, 64, 128]` |
| MTP 配置 | `with_mtp (num_speculative_tokens=1)` vs `without_mtp` |
| 指标 | `output_tokens/sec`、mean TPOT |

**输出**：CSV 结果 + 吞吐曲线，标注 crossover point（MTP 收益转为负收益的拐点），该拐点即为默认阈值建议值。

**实现方式**：基于 vLLM 已有的离线推理接口（`vllm.LLM`），避免新增依赖。

---

## 实现方案：固定阈值启停

### 设计决策

选择**固定阈值**方案（方案 A）：简单、快速落地，默认关闭（阈值=0），用户根据自身 benchmark 结果配置合适阈值。

### 实现层次

在 `MtpProposer._propose()` 入口处加早返回逻辑。选择此层的原因：
- 改动最小，不破坏调度器逻辑
- `scheduler_output` 已作为参数传入，可直接获取请求数量信息

### 改动清单

#### 1. `vllm_ascend/envs.py`

新增环境变量：

```python
"VLLM_ASCEND_MTP_DISABLE_CONCURRENCY_THRESHOLD": lambda: int(
    os.getenv("VLLM_ASCEND_MTP_DISABLE_CONCURRENCY_THRESHOLD", "0")
),
```

- `0`（默认）：不启用自适应，MTP 始终开启
- `N > 0`：当 `num_running_requests >= N` 时，禁用 MTP 投机推理

命名遵循 `VLLM_ASCEND_*` 约定。

#### 2. `vllm_ascend/spec_decode/mtp_proposer.py`

在 `MtpProposer._propose()` 方法入口（两个 code path 都需覆盖）添加：

```python
from vllm_ascend import envs

threshold = envs.VLLM_ASCEND_MTP_DISABLE_CONCURRENCY_THRESHOLD
if threshold > 0 and scheduler_output is not None:
    num_running = (
        len(scheduler_output.scheduled_new_reqs)
        + len(scheduler_output.scheduled_resumed_reqs)
        + len(scheduler_output.scheduled_cached_reqs)
    )
    if num_running >= threshold:
        batch_size = next_token_ids.shape[0]
        return torch.zeros(
            batch_size,
            self.num_speculative_tokens,
            dtype=torch.int64,
            device=next_token_ids.device,
        )
```

**注意**：该 `_propose()` 中有两条代码路径（`pcp_size * dcp_size == 1` 的 super() 调用路径 和 自定义路径），需要在两条路径的公共入口处插入，确保均生效。

#### 3. `tests/ut/spec_decode/test_mtp_proposer.py`

新增单测：
- 当 `VLLM_ASCEND_MTP_DISABLE_CONCURRENCY_THRESHOLD=N` 且 `num_running_requests >= N` 时，验证 `_propose()` 返回全零张量
- 当 `num_running_requests < N` 时，验证正常走投机推理路径

#### 4. `benchmarks/benchmark_mtp_concurrency.py`（新文件）

实现验证 benchmark 脚本，用于：
1. 离线验证 MTP 与非 MTP 在不同并发下的吞吐差异
2. 辅助用户确定合适的阈值

---

## 使用方式

```bash
# 启用自适应：并发 >= 32 时关闭 MTP
export VLLM_ASCEND_MTP_DISABLE_CONCURRENCY_THRESHOLD=32

# 先运行 benchmark 确定最优阈值
python benchmarks/benchmark_mtp_concurrency.py \
    --model deepseek-ai/DeepSeek-V3 \
    --concurrency-levels 1 2 4 8 16 32 64 128 \
    --num-speculative-tokens 1 \
    --output results/mtp_concurrency.csv
```

---

## 风险与注意事项

1. **阈值依赖硬件**：不同 NPU 卡数、模型大小的拐点不同，无通用默认值，默认 0（不启用）
2. **scheduler_output 结构变化**：如上游 vLLM 修改 `SchedulerOutput` 字段，需同步更新计数逻辑
3. **接受率不一定随并发线性下降**：实际拐点需 benchmark 确认，假设可能在某些模型上不成立

---

## 参考

- `vllm_ascend/spec_decode/mtp_proposer.py` — MTP 投机推理实现
- `vllm_ascend/core/scheduler_dynamic_batch.py` — 现有动态 batch 调度器（设计参考）
- `vllm_ascend/envs.py` — 环境变量中心化管理
- vLLM v1 Spec Decode: `vllm/v1/spec_decode/`
