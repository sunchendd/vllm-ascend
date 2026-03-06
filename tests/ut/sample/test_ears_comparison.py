#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# EARS vs Standard Rejection Sampling: Unit-Level Acceptance Rate Comparison
#
# This script simulates speculative decoding rejection sampling scenarios
# and compares acceptance rates between standard and EARS methods.

import time
from unittest.mock import patch

import torch

from vllm_ascend.sample.rejection_sampler import (
    rejection_random_sample_ears_pytorch,
    rejection_random_sample_pytorch,
)

PLACEHOLDER_TOKEN_ID = -1


def mock_pin_memory(original_func):
    def func_wo_pin_memory(*args, **kwargs):
        if kwargs.get("pin_memory", False):
            kwargs["pin_memory"] = False
        return original_func(*args, **kwargs)
    return func_wo_pin_memory


# Patch pin_memory for CPU-only testing
_patches = [
    patch("torch.arange", new=mock_pin_memory(torch.arange)),
    patch("torch.ones", new=mock_pin_memory(torch.ones)),
    patch("torch.full", new=mock_pin_memory(torch.full)),
    patch("torch.tensor", new=mock_pin_memory(torch.tensor)),
]


def start_patches():
    for p in _patches:
        p.start()


def stop_patches():
    for p in _patches:
        p.stop()


def count_accepted_tokens(output_token_ids, draft_token_ids, cu_num_draft_tokens):
    """Count how many draft tokens were accepted (not replaced by recovered tokens)."""
    batch_size = output_token_ids.shape[0]
    total_accepted = 0
    total_draft = 0

    for req_idx in range(batch_size):
        if req_idx == 0:
            start = 0
        else:
            start = cu_num_draft_tokens[req_idx - 1].item()
        end = cu_num_draft_tokens[req_idx].item()
        num_draft = end - start

        for pos in range(num_draft):
            total_draft += 1
            out_token = output_token_ids[req_idx, pos].item()
            draft_token = draft_token_ids[start + pos].item()
            if out_token == draft_token:
                total_accepted += 1

    return total_accepted, total_draft


def generate_scenario(batch_size, max_spec_len, vocab_size, uncertainty_level, seed=42):
    """Generate a test scenario with controllable uncertainty.

    Args:
        uncertainty_level: "low" (confident), "medium", "high" (uniform-ish)
    """
    torch.manual_seed(seed)
    num_tokens = batch_size * max_spec_len

    # Draft token ids
    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,))

    # Draft probs: each draft token gets a moderate probability
    draft_probs = torch.zeros(num_tokens, vocab_size)
    for i in range(num_tokens):
        draft_probs[i] = torch.softmax(torch.randn(vocab_size), dim=0)
        # Boost the drafted token's probability
        draft_probs[i, draft_token_ids[i]] = max(draft_probs[i, draft_token_ids[i]].item(), 0.3)
        draft_probs[i] = draft_probs[i] / draft_probs[i].sum()

    # Target probs: control uncertainty via temperature-like scaling
    target_logits = torch.randn(num_tokens, vocab_size)
    if uncertainty_level == "low":
        # Confident: sharp distribution
        target_probs = torch.softmax(target_logits * 5.0, dim=0)
    elif uncertainty_level == "medium":
        # Moderate uncertainty
        target_probs = torch.softmax(target_logits * 1.0, dim=0)
    else:  # high
        # High uncertainty: nearly uniform
        target_probs = torch.softmax(target_logits * 0.2, dim=0)

    # Ensure some overlap with draft tokens (realistic scenario)
    for i in range(num_tokens):
        target_probs[i, draft_token_ids[i]] += 0.05
        target_probs[i] = target_probs[i] / target_probs[i].sum()

    cu_num_draft_tokens = torch.tensor(
        [max_spec_len * (i + 1) for i in range(batch_size)]
    )

    bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1))
    recovered_token_ids = torch.randint(0, vocab_size, (num_tokens,))
    uniform_probs = torch.rand(num_tokens)
    is_greedy = torch.zeros(batch_size, dtype=torch.bool)

    return {
        "draft_token_ids": draft_token_ids,
        "draft_probs": draft_probs,
        "target_probs": target_probs,
        "cu_num_draft_tokens": cu_num_draft_tokens,
        "bonus_token_ids": bonus_token_ids,
        "recovered_token_ids": recovered_token_ids,
        "uniform_probs": uniform_probs,
        "is_greedy": is_greedy,
        "max_spec_len": max_spec_len,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
    }


def run_comparison(scenario, tolerance_values):
    """Run standard vs EARS comparison for given scenario."""
    results = {}

    batch_size = scenario["batch_size"]
    max_spec_len = scenario["max_spec_len"]
    vocab_size = scenario["vocab_size"]

    # Standard rejection sampling
    output_std = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID
    )
    t0 = time.perf_counter()
    rejection_random_sample_pytorch(
        output_std,
        scenario["cu_num_draft_tokens"],
        scenario["draft_token_ids"],
        scenario["draft_probs"],
        scenario["target_probs"],
        scenario["bonus_token_ids"],
        scenario["recovered_token_ids"],
        scenario["uniform_probs"],
        scenario["is_greedy"],
        max_spec_len,
        vocab_size,
    )
    t1 = time.perf_counter()
    accepted_std, total_std = count_accepted_tokens(
        output_std, scenario["draft_token_ids"], scenario["cu_num_draft_tokens"]
    )
    results["standard"] = {
        "accepted": accepted_std,
        "total": total_std,
        "rate": accepted_std / total_std if total_std > 0 else 0,
        "time_ms": (t1 - t0) * 1000,
    }

    # EARS with different tolerances
    for tol in tolerance_values:
        output_ears = torch.full(
            (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID
        )
        t0 = time.perf_counter()
        rejection_random_sample_ears_pytorch(
            output_ears,
            scenario["cu_num_draft_tokens"],
            scenario["draft_token_ids"],
            scenario["draft_probs"],
            scenario["target_probs"],
            scenario["bonus_token_ids"],
            scenario["recovered_token_ids"],
            scenario["uniform_probs"],
            scenario["is_greedy"],
            max_spec_len,
            vocab_size,
            base_tolerance=tol,
        )
        t1 = time.perf_counter()
        accepted_ears, total_ears = count_accepted_tokens(
            output_ears, scenario["draft_token_ids"], scenario["cu_num_draft_tokens"]
        )
        results[f"ears_tol={tol}"] = {
            "accepted": accepted_ears,
            "total": total_ears,
            "rate": accepted_ears / total_ears if total_ears > 0 else 0,
            "time_ms": (t1 - t0) * 1000,
        }

    return results


def run_multi_seed_comparison(batch_size, max_spec_len, vocab_size,
                              uncertainty_level, tolerance_values, num_seeds=20):
    """Run comparison across multiple random seeds for statistical robustness."""
    all_results = {}
    for seed in range(num_seeds):
        scenario = generate_scenario(
            batch_size, max_spec_len, vocab_size, uncertainty_level, seed=seed
        )
        results = run_comparison(scenario, tolerance_values)
        for method, data in results.items():
            if method not in all_results:
                all_results[method] = {"rates": [], "times": []}
            all_results[method]["rates"].append(data["rate"])
            all_results[method]["times"].append(data["time_ms"])

    # Compute statistics
    stats = {}
    for method, data in all_results.items():
        rates = torch.tensor(data["rates"])
        times = torch.tensor(data["times"])
        stats[method] = {
            "mean_rate": rates.mean().item(),
            "std_rate": rates.std().item(),
            "min_rate": rates.min().item(),
            "max_rate": rates.max().item(),
            "mean_time_ms": times.mean().item(),
        }
    return stats


def main():
    start_patches()

    print("=" * 80)
    print("EARS vs Standard Rejection Sampling: Acceptance Rate Comparison")
    print("=" * 80)

    tolerance_values = [0.05, 0.1, 0.2, 0.3, 0.5]
    configs = [
        # (batch_size, max_spec_len, vocab_size, uncertainty, label)
        (16, 3, 1024, "high", "High Uncertainty (batch=16, spec_len=3)"),
        (16, 3, 1024, "medium", "Medium Uncertainty (batch=16, spec_len=3)"),
        (16, 3, 1024, "low", "Low Uncertainty (batch=16, spec_len=3)"),
        (32, 1, 1024, "high", "High Uncertainty (batch=32, spec_len=1, MTP-like)"),
        (32, 1, 1024, "medium", "Medium Uncertainty (batch=32, spec_len=1, MTP-like)"),
        (8, 5, 2048, "high", "High Uncertainty (batch=8, spec_len=5, large vocab)"),
        (8, 5, 2048, "medium", "Medium Uncertainty (batch=8, spec_len=5, large vocab)"),
    ]

    all_config_results = []

    for batch_size, max_spec_len, vocab_size, uncertainty, label in configs:
        print(f"\n{'─' * 70}")
        print(f"Scenario: {label}")
        print(f"  batch_size={batch_size}, max_spec_len={max_spec_len}, "
              f"vocab_size={vocab_size}, uncertainty={uncertainty}")
        print(f"{'─' * 70}")

        stats = run_multi_seed_comparison(
            batch_size, max_spec_len, vocab_size, uncertainty,
            tolerance_values, num_seeds=20
        )

        std_rate = stats["standard"]["mean_rate"]
        print(f"  {'Method':<25} {'Accept Rate':>12} {'±Std':>8} {'Δ vs Std':>10} {'Time(ms)':>10}")
        print(f"  {'─' * 65}")

        for method, s in sorted(stats.items()):
            delta = s["mean_rate"] - std_rate
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            if method == "standard":
                delta_str = "baseline"
            print(f"  {method:<25} {s['mean_rate']:>12.4f} {s['std_rate']:>7.4f} "
                  f"{delta_str:>10} {s['mean_time_ms']:>9.3f}")

        all_config_results.append({
            "label": label,
            "uncertainty": uncertainty,
            "batch_size": batch_size,
            "max_spec_len": max_spec_len,
            "stats": stats,
        })

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: Acceptance Rate Improvement (EARS tolerance=0.1 vs Standard)")
    print(f"{'=' * 80}")
    print(f"  {'Scenario':<55} {'Standard':>10} {'EARS 0.1':>10} {'Δ':>8}")
    print(f"  {'─' * 83}")
    for r in all_config_results:
        std = r["stats"]["standard"]["mean_rate"]
        ears = r["stats"].get("ears_tol=0.1", {}).get("mean_rate", 0)
        delta = ears - std
        print(f"  {r['label']:<55} {std:>10.4f} {ears:>10.4f} {delta:>+8.4f}")

    stop_patches()
    print("\nUnit-level comparison complete.")


if __name__ == "__main__":
    main()
