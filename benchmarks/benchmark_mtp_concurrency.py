#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Benchmark MTP speculative decoding performance across concurrency levels.

This script validates the hypothesis that MTP speculative decoding improves
throughput at low concurrency but degrades it at high concurrency on Ascend NPU.
The crossover point it identifies can be used as the value for:

    VLLM_ASCEND_MTP_DISABLE_CONCURRENCY_THRESHOLD

Usage:
    python benchmarks/benchmark_mtp_concurrency.py \\
        --model /path/to/deepseek-v3 \\
        --concurrency-levels 1 2 4 8 16 32 64 \\
        --num-speculative-tokens 1 \\
        --output-len 128 \\
        --output results/mtp_concurrency_bench.csv
"""

import argparse
import csv
import gc
import os
import time
from collections import defaultdict


def run_batch(model_path: str, batch_size: int, output_len: int,
              num_spec_tokens: int, tensor_parallel_size: int) -> float:
    """Run offline inference and return output tokens/sec.

    Args:
        model_path: Path to the model.
        batch_size: Number of concurrent requests to simulate.
        output_len: Number of output tokens per request.
        num_spec_tokens: Number of MTP speculative tokens (0 = disabled).
        tensor_parallel_size: Number of NPU devices for tensor parallelism.

    Returns:
        Output throughput in tokens/sec.
    """
    from vllm import LLM, SamplingParams

    kwargs: dict = {
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": 2048,
        "enforce_eager": True,  # disable graph capture for fair comparison
    }
    if num_spec_tokens > 0:
        kwargs["speculative_config"] = {
            "method": "deepseek_mtp",
            "num_speculative_tokens": num_spec_tokens,
        }

    llm = LLM(model=model_path, **kwargs)
    prompts = ["Tell me about artificial intelligence."] * batch_size
    params = SamplingParams(temperature=0.0, max_tokens=output_len)

    # Warm-up
    llm.generate(prompts[:1], params)

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    del llm
    gc.collect()
    try:
        import torch
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
    except Exception:
        pass

    return total_tokens / elapsed if elapsed > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MTP vs baseline throughput across concurrency levels"
    )
    parser.add_argument("--model", required=True,
                        help="Path to DeepSeek-V3/R1 model")
    parser.add_argument("--concurrency-levels", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32, 64],
                        help="List of batch sizes to test")
    parser.add_argument("--num-speculative-tokens", type=int, default=1,
                        help="Number of MTP speculative tokens to use")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Number of output tokens per request")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (number of NPU devices)")
    parser.add_argument("--output", default="mtp_concurrency_bench.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    rows = []
    for concurrency in sorted(args.concurrency_levels):
        for label, spec in [("baseline", 0),
                             ("mtp", args.num_speculative_tokens)]:
            print(f"[{label}] concurrency={concurrency} ...", flush=True)
            tps = run_batch(
                model_path=args.model,
                batch_size=concurrency,
                output_len=args.output_len,
                num_spec_tokens=spec,
                tensor_parallel_size=args.tensor_parallel_size,
            )
            print(f"  → {tps:.1f} tokens/sec", flush=True)
            rows.append({
                "concurrency": concurrency,
                "mode": label,
                "tokens_per_sec": round(tps, 2),
            })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["concurrency", "mode", "tokens_per_sec"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {args.output}")

    # Summary table with crossover detection
    by_c: dict = defaultdict(dict)
    for r in rows:
        by_c[r["concurrency"]][r["mode"]] = r["tokens_per_sec"]

    print("\n{:>12} | {:>10} | {:>10} | {:>10}".format(
        "Concurrency", "Baseline", "MTP", "MTP_gain%"))
    print("-" * 52)
    crossover = None
    for c in sorted(by_c):
        b = by_c[c].get("baseline", 0.0)
        m = by_c[c].get("mtp", 0.0)
        gain = (m - b) / b * 100 if b > 0 else float("nan")
        marker = ""
        if gain < 0 and crossover is None:
            crossover = c
            marker = "  ← crossover point"
        print(f"{c:>12} | {b:>10.1f} | {m:>10.1f} | {gain:>+9.1f}%{marker}")

    if crossover is not None:
        print(f"\nSuggested threshold: VLLM_ASCEND_MTP_DISABLE_CONCURRENCY_THRESHOLD={crossover}")
    else:
        print("\nNo crossover detected: MTP is beneficial at all tested concurrency levels.")


if __name__ == "__main__":
    main()
