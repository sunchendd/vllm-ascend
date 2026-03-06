#!/usr/bin/env python3
"""
E2E Benchmark: EARS vs Standard Rejection Sampling on DeepSeek-R1-0528-W8A8
Measures throughput (tokens/sec) and output token acceptance for MTP spec decode.
"""
import argparse
import json
import time

import requests

PROMPTS = [
    "Explain the concept of entropy in information theory in 100 words.",
    "Write a Python function to compute fibonacci numbers efficiently.",
    "What is speculative decoding and how does it improve LLM inference speed?",
    "Describe the difference between greedy decoding and sampling with temperature.",
    "Solve this math problem step by step: If f(x) = 3x^2 + 2x - 1, find f'(x).",
    "Explain the transformer attention mechanism in simple terms.",
    "What are the advantages of quantization (W8A8) for large language models?",
    "Write a brief comparison of MoE (Mixture of Experts) vs dense models.",
]


def run_benchmark(base_url, model_name, num_requests=8, max_tokens=256):
    """Send requests to vLLM server and measure throughput."""
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0

    print(f"  Running {num_requests} requests (max_tokens={max_tokens})...")

    for i in range(num_requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            t1 = time.perf_counter()

            if resp.status_code != 200:
                print(f"    Request {i+1} failed: HTTP {resp.status_code}")
                print(f"    Response: {resp.text[:200]}")
                continue

            data = resp.json()
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            elapsed = t1 - t0

            results.append({
                "request_id": i + 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_s": elapsed,
                "tokens_per_sec": completion_tokens / elapsed if elapsed > 0 else 0,
            })

            total_input_tokens += prompt_tokens
            total_output_tokens += completion_tokens
            total_time += elapsed

            print(f"    Request {i+1}: {completion_tokens} tokens in {elapsed:.2f}s "
                  f"({completion_tokens/elapsed:.1f} tok/s)")
        except Exception as e:
            print(f"    Request {i+1} error: {e}")

    if not results:
        return None

    avg_tps = total_output_tokens / total_time if total_time > 0 else 0
    return {
        "num_requests": len(results),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_time_s": total_time,
        "avg_tokens_per_sec": avg_tps,
        "avg_latency_s": total_time / len(results),
        "per_request": results,
    }


def check_server_health(base_url, timeout=10):
    """Check if vLLM server is ready."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def get_model_name(base_url):
    """Get model name from vLLM server."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        data = resp.json()
        return data["data"][0]["id"]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="EARS E2E Benchmark")
    parser.add_argument("--base-url", default="http://localhost:8100",
                        help="vLLM server base URL")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--label", default="benchmark",
                        help="Label for this run (e.g., 'standard' or 'ears')")
    parser.add_argument("--output-file", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"E2E Benchmark: {args.label}")
    print(f"Server: {args.base_url}")
    print(f"{'='*60}")

    # Check server health
    if not check_server_health(args.base_url):
        print("ERROR: Server is not healthy or not reachable.")
        return

    model_name = get_model_name(args.base_url)
    if not model_name:
        print("ERROR: Could not get model name from server.")
        return
    print(f"Model: {model_name}")

    # Warmup
    print("\nWarmup (2 requests)...")
    run_benchmark(args.base_url, model_name, num_requests=2, max_tokens=64)

    # Actual benchmark
    print(f"\nBenchmark ({args.num_requests} requests, max_tokens={args.max_tokens})...")
    results = run_benchmark(
        args.base_url, model_name,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
    )

    if results:
        print(f"\n{'─'*50}")
        print(f"Results ({args.label}):")
        print(f"  Total requests:      {results['num_requests']}")
        print(f"  Total output tokens: {results['total_output_tokens']}")
        print(f"  Total time:          {results['total_time_s']:.2f}s")
        print(f"  Avg tokens/sec:      {results['avg_tokens_per_sec']:.2f}")
        print(f"  Avg latency:         {results['avg_latency_s']:.2f}s")

        if args.output_file:
            results["label"] = args.label
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
