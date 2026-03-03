# Draft Model Position OOB Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify that `0011-xfusion-vLLM-ascend-fix-draft-model-position-oob.patch` correctly prevents the fatal Ascend CANN crash when a speculative-decoding draft model has `max_position_embeddings` smaller than the target model's context.

**Architecture:** The fix stores `draft_max_model_len` from `draft_model_config` in `DraftModelProposer.__init__` and replaces all three locations that used the target model's `max_model_len` for position clamping. Positions ≥ draft limit are clamped to 0; their KV-cache slot_mapping is masked with `PADDING_SLOT_ID` so no KV state is written.

**Tech Stack:** Python 3.11, PyTorch, vllm-ascend 0.12.0rc1, Ascend CANN 8.3.RC2, `patch` (unified diff)

---

### Task 1: Apply all patches to source tree

**Files:**
- Run: `vLLM-ascend/apply-patch.sh -d /tmp/vllm-ascend-src`

**Step 1: Download and patch the source**

```bash
cd /workspace/vllm-ascend/vLLM-ascend
bash apply-patch.sh -d /tmp/vllm-ascend-src
```

Expected: `All patches applied successfully.`  
If the internal artifactory is unreachable, clone vllm-ascend from GitHub:
```bash
pip install vllm-ascend==0.12.0rc1
```

**Step 2: Verify patch 0011 was applied**

```bash
grep -n 'draft_max_model_len' /tmp/vllm-ascend-src/vllm_ascend/spec_decode/draft_model_proposer.py
```

Expected output (4 lines):
```
134:        self.draft_max_model_len = vllm_config.speculative_config.draft_model_config.max_model_len
265:            max_model_len=self.draft_max_model_len,
437:        _exceeds_draft = target_positions >= self.draft_max_model_len
554:            exceeds_max_model_len = positions_cpu >= self.draft_max_model_len
```

---

### Task 2: Write unit test for position clamping logic

**Files:**
- Create: `/tmp/vllm-ascend-src/tests/test_draft_position_clamp.py`

**Step 1: Write the test**

```python
"""
Test that DraftModelProposer clamps positions exceeding draft_max_model_len.
Validates the fix for CANN gather_v3 OOB crash (patch 0011).
"""
import torch
import pytest
from unittest.mock import MagicMock, patch

PADDING_SLOT_ID = -1
DRAFT_MAX = 40960
TARGET_MAX = 131072


def _make_mock_config(draft_max=DRAFT_MAX, target_max=TARGET_MAX):
    cfg = MagicMock()
    cfg.model_config.max_model_len = target_max
    cfg.model_config.dtype = torch.float16
    cfg.speculative_config.draft_model_config.max_model_len = draft_max
    cfg.speculative_config.draft_model_config.get_hidden_size.return_value = 1024
    cfg.scheduler_config.max_num_seqs = 32
    cfg.scheduler_config.max_num_batched_tokens = 2048
    cfg.cache_config.block_size = 16
    cfg.compilation_config.mode = MagicMock()
    cfg.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    return cfg


def test_positions_below_draft_max_are_unchanged():
    """Positions within draft model range must not be modified."""
    positions = torch.tensor([0, 100, 40959], dtype=torch.long)
    slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    exceeds = positions >= DRAFT_MAX
    assert not exceeds.any()
    # slot_mapping must stay intact
    assert (slot_mapping == torch.tensor([0, 1, 2])).all()


def test_positions_at_draft_max_are_clamped():
    """Position == draft_max_model_len must be clamped to 0."""
    positions = torch.tensor([40958, 40959, 40960, 65535], dtype=torch.long)
    slot_mapping = torch.tensor([100, 200, 300, 400], dtype=torch.int32)

    exceeds = positions >= DRAFT_MAX
    clamped = positions.clone()
    clamped[exceeds] = 0
    masked_slots = slot_mapping.masked_fill(exceeds, PADDING_SLOT_ID)

    assert clamped[0] == 40958
    assert clamped[1] == 40959
    assert clamped[2] == 0      # 40960 → clamped
    assert clamped[3] == 0      # 65535 → clamped
    assert masked_slots[0] == 100
    assert masked_slots[1] == 200
    assert masked_slots[2] == PADDING_SLOT_ID  # masked
    assert masked_slots[3] == PADDING_SLOT_ID  # masked


def test_seq_lens_use_original_positions():
    """
    seq_lens must reflect the true (unclamped) sequence length so that
    the attention kernel attends to the correct number of KV-cache entries.
    """
    # Simulate: one request with 65536 tokens (last position = 65535)
    last_token_positions = torch.tensor([65535], dtype=torch.long)
    seq_lens = (last_token_positions + 1).int()
    assert seq_lens[0] == 65536, "seq_lens must use original position"

    # After clamping, seq_lens must NOT change
    clamped = last_token_positions.clone()
    clamped[clamped >= DRAFT_MAX] = 0
    seq_lens_after_clamp = (last_token_positions + 1).int()  # still from original
    assert seq_lens_after_clamp[0] == 65536


def test_no_crash_when_all_positions_in_range():
    """No-op: when all positions < draft_max, nothing is modified."""
    positions = torch.arange(0, 100, dtype=torch.long)
    slot_mapping = torch.arange(0, 100, dtype=torch.int32)
    exceeds = positions >= DRAFT_MAX
    assert not exceeds.any()
    # Neither positions nor slot_mapping changed
    assert (positions == torch.arange(0, 100, dtype=torch.long)).all()
```

**Step 2: Run the tests**

```bash
cd /tmp/vllm-ascend-src
python -m pytest tests/test_draft_position_clamp.py -v
```

Expected:
```
PASSED tests/test_draft_position_clamp.py::test_positions_below_draft_max_are_unchanged
PASSED tests/test_draft_position_clamp.py::test_positions_at_draft_max_are_clamped
PASSED tests/test_draft_position_clamp.py::test_seq_lens_use_original_positions
PASSED tests/test_draft_position_clamp.py::test_no_crash_when_all_positions_in_range
4 passed in 0.xx s
```

**Step 3: Commit the test**

```bash
cd /workspace/vllm-ascend
git add /tmp/vllm-ascend-src/tests/test_draft_position_clamp.py
# (or copy into the patch repo for inclusion)
git commit -m "test: unit tests for draft model position OOB clamping logic"
```

---

### Task 3: Integration smoke test (requires Ascend NPU environment)

This step requires the actual Ascend NPU hardware and the running service. Skip on CPU-only machines.

**Step 1: Start the inference service with draft model**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --speculative-model Qwen/Qwen3-0.6B \
  --num-speculative-tokens 5 \
  --tensor-parallel-size 4 \
  --max-model-len 131072
```

**Step 2: Send a request that exceeds draft model's limit (64K)**

```bash
python - << 'EOF'
import requests, json
# Generate a prompt of ~40K tokens
prompt = "Hello " * 22000  # ~40K tokens

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
    },
    timeout=300
)
print(f"Status: {resp.status_code}")
data = resp.json()
if "error" in data:
    print(f"Error: {data['error']}")
else:
    print(f"Response OK, tokens: {data['usage']}")
EOF
```

**Expected (BEFORE fix):** Service crashes with `RuntimeError: gather_v3 Index 40960 out of range`  
**Expected (AFTER fix):** `Status: 200` — service returns a response and remains alive

**Step 3: Verify service is still alive after the long-context request**

```bash
curl http://localhost:8000/v1/models
# Expected: 200 OK with model list
```

---

### Task 4: Code review

Run the code reviewer agent:

```bash
# In Copilot session
# Use the superpowers/code-reviewer agent on the changes in patch 0011
```

Verify:
- No other locations in `draft_model_proposer.py` use target `max_model_len` for position clamping
- `seq_lens` is correctly computed from original positions in all paths
- The `PADDING_SLOT_ID` masking prevents KV cache corruption for out-of-range positions


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
