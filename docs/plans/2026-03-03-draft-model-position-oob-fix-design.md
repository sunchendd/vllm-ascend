# Design: Fix Draft Model Position Out-of-Bounds Crash in Speculative Decoding

## Problem

When using speculative decoding with a draft model whose `max_position_embeddings` is smaller than the target model's `max_model_len`, requests with long context cause the inference service to crash rather than gracefully degrading.

**Scenario**: Qwen3-30B-A3B-Instruct (target, 128K context) + Qwen3-0.6B (draft, 40960 max positions). A 64K context request causes a fatal Ascend CANN NPU error:

```
[ASSERT] gather_v3_base.h:137: Assertion `(0 <= val && val < this->gxSize_)' Index 40960 out of range[0 40960)!
RuntimeError: The Inner error is reported above. current working operator: aclnnMatmulWeightNz
```

The process exits, taking down the entire inference service.

## Root Cause

In `vllm_ascend/spec_decode/draft_model_proposer.py`, all position clamping uses `self.vllm_config.model_config.max_model_len` (the **target** model's limit, e.g. 128K). The draft model's `max_model_len` (40960) is never consulted.

Three code paths pass un-clamped positions to the draft model:

| Location | Issue |
|---|---|
| `__init__` | No `draft_max_model_len` stored |
| `generate_token_ids` → `merge_next_token_ids_into_token_ids` | `max_model_len` arg uses target model limit |
| `_propose` (initial prefill pass) | `target_positions` passed directly to draft model without clamping |
| `_propose` (decode loop) | `exceeds_max_model_len` checks against target model limit |

When position ID 40960 reaches the draft model's position embedding table (size=40960, valid range [0, 40959)), CANN's `gather_v3` operator triggers a fatal assertion.

## Design

### Approach (chosen)

Clamp position IDs to `[0, draft_max_model_len - 1]` before passing to the draft model. Mask the corresponding KV-cache `slot_mapping` entries with `PADDING_SLOT_ID` so no KV state is written for out-of-range positions.

**Key insight**: `seq_lens` must be computed from the **original** (unclamped) positions. The attention kernel uses `seq_lens` to know how many KV-cache entries to attend to — this must reflect the true sequence length. Only the position IDs used by RoPE need clamping.

Draft tokens produced from clamped (garbage) positions will be rejected at the target model's verification step. The service continues without crashing.

### Alternatives Considered

| Approach | Trade-off |
|---|---|
| **Skip speculative decoding for over-limit requests** | Clean semantics, complex to implement per-request skipping in current architecture |
| **Clamp positions (chosen)** | Minimal code change, safe — bad draft tokens are always rejected |
| **Error at startup if draft max < target max** | Fails legitimate mixed-length use cases (short requests would work fine) |

### Changes

File: `vllm_ascend/spec_decode/draft_model_proposer.py`

1. **`__init__`**: Store `self.draft_max_model_len = vllm_config.speculative_config.draft_model_config.max_model_len`

2. **`generate_token_ids`**: Pass `max_model_len=self.draft_max_model_len` to `merge_next_token_ids_into_token_ids` (masks KV slot_mapping for positions ≥ draft limit)

3. **`_propose` (initial prefill)**:  After `seq_lens` is computed from original positions, clamp `target_positions` and mask `target_slot_mapping` for any positions ≥ `draft_max_model_len`

4. **`_propose` (decode loop)**: Replace `self.vllm_config.model_config.max_model_len` with `self.draft_max_model_len` in the `exceeds_max_model_len` check

### Delivery

New patch: `vllm-ascend/patch/0011-xfusion-vLLM-ascend-fix-draft-model-position-oob.patch`  
Added to: `vllm-ascend/series.conf`

No changes to vLLM upstream patches needed (the analogous GPU path in `vllm/v1/spec_decode/draft_model.py` has the same issue but is out of scope).
