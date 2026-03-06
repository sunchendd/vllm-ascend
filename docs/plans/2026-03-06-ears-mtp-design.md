# EARS for DeepSeek-MTP Design

## Problem

The EARS (Entropy-Adaptive Rejection Sampling) algorithm improves speculative decoding acceptance rates by dynamically adjusting the rejection threshold based on prediction uncertainty. Two existing patches (0004, 0007) implement EARS for eagle3/suffix methods on an older codebase with `AscendRejectionSampler`. The current codebase uses upstream `RejectionSampler` with function-level monkey-patching, so these patches cannot be applied directly. We need to implement EARS for DeepSeek-MTP (and eagle3/suffix) on the current architecture.

## Approach

**Subclass `RejectionSampler`** to create `EntropyAdaptiveRejectionSampler` that overrides `forward()` with EARS logic. Conditionally instantiate it in the V1 model runner based on environment variable and spec decode method.

## Architecture

### Components

```
envs.py                          → VLLM_EARS_TOLERANCE env var (default 0.0)
sample/rejection_sampler.py      → EntropyAdaptiveRejectionSampler class
                                   ears_rejection_sample() function
                                   rejection_random_sample_ears_pytorch() function
worker/model_runner_v1.py        → Conditional sampler instantiation in _set_up_drafter()
tests/ut/sample/test_ears.py     → Unit tests for EARS logic
```

### Data Flow

```
target_logits
  → apply_sampling_constraints() [existing, unchanged]
  → softmax → target_probs
  → ears_rejection_sample():
      → greedy path: unchanged (argmax comparison)
      → random path: rejection_random_sample_ears_pytorch()
          → compute uncertainty = 1 - max(target_probs) per token
          → tolerance = base_tolerance × uncertainty
          → accept if target_prob/draft_prob >= (uniform_prob - tolerance)
  → output_token_ids
```

### EARS Algorithm

The core modification is in the random sampling rejection step:

1. Compute `max_target_probs = target_probs.max(dim=-1).values`
2. Compute `uncertainties = 1.0 - max_target_probs`
3. For each draft token where `target_prob / draft_prob < uniform_prob`:
   - `tolerance = base_tolerance × uncertainty`
   - If `target_prob / draft_prob >= (uniform_prob - tolerance)`: **accept** (would normally reject)
   - Otherwise: reject as normal

This means tokens where the model is uncertain get a higher acceptance tolerance.

### Environment Variable

- Name: `VLLM_EARS_TOLERANCE`
- Type: float
- Default: `0.0` (disabled)
- Range: 0.0 to 1.0
- When > 0 and method is mtp/eagle3/suffix: enables EARS

### Supported Methods

- `mtp` (primary target: DeepSeek-MTP)
- `eagle3` (original patch target)
- `suffix` (from patch 0007)

### Files Changed

1. **`vllm_ascend/envs.py`** — Add `VLLM_EARS_TOLERANCE`
2. **`vllm_ascend/sample/rejection_sampler.py`** — Add EARS class and functions
3. **`vllm_ascend/worker/model_runner_v1.py`** — Conditional sampler selection
4. **`tests/ut/sample/test_ears.py`** — Unit tests

### Testing

- Unit test: Verify EARS acceptance rate > standard for high-uncertainty scenarios
- Unit test: Verify EARS matches standard behavior when tolerance = 0
- Unit test: Verify greedy mode unaffected by EARS

## Notes

- V2 model runner is NOT modified (no MTP support in V2)
- The `ears_rejection_sample` function reuses existing helper functions (greedy sampling, recovered token sampling) and only modifies the random sampling path
- Block verify path (max_spec_len >= 3) in EARS uses the simpler pytorch path to avoid Triton kernel changes
