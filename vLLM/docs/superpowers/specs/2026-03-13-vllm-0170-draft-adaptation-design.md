# vLLM 0.17.0 Draft Adaptation Design

## Problem Statement

Adapt the local xfusion speculative decoding work to `vLLM 0.17.0` with the
following boundaries:

- The runtime code that must actually take effect lives under
  `/usr/local/lib/python3.12/dist-packages/vllm`.
- Upstream `0.17.0` already supports the open-source `draft_model` path, so the
  adaptation should treat that as the baseline instead of re-porting older
  draft-mode enablement wholesale.
- Add explicit support for `suffix-draft_model`.
- Support adaptive draft length only for pure `draft_model`.
- Keep `suffix-draft_model` on fixed per-method token counts.
- Validate on the current machine, where `nvidia-smi` can expose usable GPUs and
  the intended target is a two-GPU `Qwen3-32B` run on RTX 5090 cards.
- Prefer existing weights under `/data` and `/data/models`; additional assets
  may also be stored there if needed.

The goal of this work is functional bring-up first: after patching the installed
`vllm` package, the system should be able to parse the new speculative-decoding
configuration, initialize the correct proposers, and complete inference for both
pure `draft_model` with adaptive length and `suffix-draft_model` with fixed
per-method token counts.

## Proposed Approach

Use an incremental, `0.17.0`-native adaptation strategy:

1. Preserve the upstream `0.17.0` draft-model architecture as the mainline.
2. Add the missing configuration and routing needed for
   `suffix-draft_model`.
3. Introduce adaptive draft-length selection only into the pure
   `draft_model` execution path.
4. Carry over only the optimization hooks that are required to make the above
   work safely on `0.17.0`, especially metrics-based length selection,
   cudagraph routing exclusions, confidence-gated drafting, and the known
   long-context draft-model safety fix from the local branch.

This minimizes divergence from upstream `0.17.0`, keeps the design aligned with
current file boundaries, and reduces the risk of reviving obsolete assumptions
from older patch generations.

## Requirements

### Functional Requirements

1. The speculative configuration must accept a new explicit method:
   `suffix-draft_model`.
2. The configuration must support
   `num_speculative_tokens_per_method: dict[str, int]` for combination modes,
   with at least `suffix` and `draft_model` entries for
   `suffix-draft_model`.
3. Pure `draft_model` must optionally support an adaptive draft-length range via
   `speculative_token_range`.
4. `suffix-draft_model` must remain fixed-length and must not consume
   `speculative_token_range`.
5. The runtime must instantiate and invoke both suffix and draft proposers for
   `suffix-draft_model`, with suffix results preferred and draft-model fallback
   used when suffix speculation does not supply tokens.
6. The pure `draft_model` path must continue to use a single
   `DraftModelProposer`, with the selected draft length computed dynamically
   from recent acceptance behavior.
7. The modified package must work from the installed path under
   `/usr/local/lib/python3.12/dist-packages/vllm`.

### Non-Functional Requirements

1. Prioritize safe functional adaptation over full optimization parity with all
   historical patches.
2. Avoid introducing interface ambiguity between fixed-length combination modes
   and adaptive-length pure draft mode.
3. Preserve compatibility with the existing `0.17.0` proposer and scheduler
   structure.
4. Keep validation centered on local bring-up and successful inference rather
   than performance targets.

## Scope

### In Scope

- Explicit `suffix-draft_model` configuration support.
- Per-method speculative token-count validation for combination modes.
- Adaptive draft-length control for pure `draft_model`.
- Required metrics plumbing for adaptive length selection.
- Required proposer API updates to accept an explicit draft length.
- Required cudagraph dispatch adjustments for dynamic-length draft execution.
- Confidence-based early-stop integration only where needed for safe draft-path
  execution on the adapted flow.
- Relevant long-context draft-model correctness fix from the local branch.

### Out of Scope

- General speculative-decoding redesign beyond the affected draft and suffix
  flows.
- New user-facing performance commitments.
- Broad refactors unrelated to `draft_model`, `suffix`, or the selected
  combination mode.
- Optimization work for additional speculative methods beyond what is needed to
  keep shared code paths correct.

## Architecture

### 1. Configuration Layer

File focus: `vllm/config/speculative.py`

Responsibilities:

- Add `suffix-draft_model` to the speculative method surface.
- Add `num_speculative_tokens_per_method`.
- Add `speculative_token_range` for pure `draft_model`.
- Expand helper predicates such as `uses_draft_model()` and `use_ngram()` so
  combination modes are routed through the correct runtime branches.
- Enforce hard validation rules:
  - `suffix-draft_model` requires `num_speculative_tokens_per_method`.
  - `suffix-draft_model` requires both `suffix` and `draft_model` entries.
  - `suffix-draft_model` rejects `speculative_token_range`.
  - Pure `draft_model` may use `speculative_token_range`, but only with valid
    positive integer values.

The configuration layer remains the single point that defines whether the
execution path is fixed-length combination mode or adaptive-length pure draft
mode.

### 2. Runtime Routing Layer

File focus: `vllm/v1/worker/gpu_model_runner.py`

Responsibilities:

- Keep the existing pure `draft_model` path intact, still backed by one
  `DraftModelProposer`.
- Add combination-mode routing for `suffix-draft_model`, where:
  - a suffix proposer is instantiated for suffix speculation;
  - a draft-model proposer is instantiated for fallback speculation;
  - suffix output is attempted first;
  - draft-model speculation is used only when suffix speculation does not fully
    satisfy the proposal path.
- Track speculative-decoding acceptance statistics for pure `draft_model`.
- Compute and pass the next draft length into the draft proposer only for pure
  `draft_model`.
- Keep `suffix-draft_model` on fixed token counts derived from
  `num_speculative_tokens_per_method`.

This preserves a clean separation:

- `draft_model` => adaptive single-proposer path
- `suffix-draft_model` => fixed-length dual-proposer path

### 3. Draft Proposer Layer

File focus: `vllm/v1/spec_decode/draft_model.py`

Responsibilities:

- Extend proposer entry points to accept an explicit `draft_length`.
- Use the provided draft length in place of always consuming the configured
  maximum speculative-token count.
- Keep confidence-based early-stop handling internal to the proposer.
- Ensure stopped sequences do not corrupt KV state by applying the appropriate
  mask and slot-mapping behavior.
- Preserve compatibility with existing draft-model loading and model-tagging
  behavior in `0.17.0`.

### 4. Metrics Layer

File focus: `vllm/v1/spec_decode/metrics.py`

Responsibilities:

- Record aggregate acceptance behavior for pure `draft_model`.
- Maintain a smoothed acceptance signal suitable for selecting the next draft
  length from `speculative_token_range`.
- Expose a deterministic helper for computing the next draft length.

Combination modes do not consume this adaptive control. Their token counts stay
fixed by configuration.

### 5. Graph Dispatch Layer

File focus: `vllm/v1/cudagraph_dispatcher.py`

Responsibilities:

- Prevent dynamic-length draft execution from being routed into incompatible
  uniform-decode cudagraph assumptions.
- Ensure the adapted speculative methods take the safe graph-selection path for
  their runtime behavior.

## Detailed Data Flow

### Pure `draft_model` with Adaptive Length

1. User config selects `method="draft_model"` and may provide
   `speculative_token_range`.
2. `SpeculativeConfig` validates the range and normalizes any derived values.
3. `GPUModelRunner` instantiates a single `DraftModelProposer`.
4. After each speculative iteration, acceptance statistics are observed and
   stored in speculative metrics.
5. Metrics logic selects the next draft length from the configured range.
6. `GPUModelRunner` passes that explicit `draft_length` into the draft proposer.
7. `DraftModelProposer` generates up to that length, with confidence-based early
   stopping allowed to stop sooner.
8. KV writes for stopped sequences remain masked to avoid cache corruption.

### `suffix-draft_model` with Fixed Per-Method Counts

1. User config selects `method="suffix-draft_model"` and must provide
   `num_speculative_tokens_per_method`.
2. `SpeculativeConfig` verifies both `suffix` and `draft_model` entries exist.
3. `GPUModelRunner` instantiates both a suffix proposer and a draft-model
   proposer.
4. At proposal time, suffix speculation runs first.
5. If suffix speculation yields usable speculative tokens, those are used.
6. If suffix speculation is absent or insufficient for the request, the runtime
   falls back to draft-model speculation using the fixed token count defined for
   `draft_model`.
7. No adaptive draft-length selection is performed in this mode.

## Error Handling and Safety Rules

1. Invalid `suffix-draft_model` configurations must fail at config-validation
   time with explicit error messages.
2. Mixed fixed-length and adaptive-length controls must be rejected rather than
   silently ignored.
3. Confidence-gated early stopping must remain an internal optimization; it must
   not require new user-visible control flow outside the speculative config.
4. Sequence-length, slot-mapping, and KV-write behavior must remain correct for
   partially stopped draft batches.
5. The long-context draft-model position safety fix from the local branch must
   be incorporated where required so long prompt or decode states do not produce
   out-of-bounds positions for `Qwen3-32B` or similar models.

## Planned File Touch Points

Primary files expected to change:

- `vllm/config/speculative.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/spec_decode/draft_model.py`
- `vllm/v1/spec_decode/metrics.py`
- `vllm/v1/cudagraph_dispatcher.py`

Secondary files that may need change depending on exact `0.17.0` interfaces:

- `vllm/v1/spec_decode/eagle.py`
- `vllm/v1/worker/utils.py`
- dependency/config surfaces related to suffix decoding if the installed
  environment is missing required package support

## Validation Strategy

### 1. Static and Startup Validation

- Confirm the modified installed package imports successfully.
- Confirm `SpeculativeConfig` parses:
  - pure `draft_model` with `speculative_token_range`
  - `suffix-draft_model` with `num_speculative_tokens_per_method`
- Confirm proposer initialization selects the intended runtime objects.

### 2. Functional Validation

- Run at least one inference path for pure `draft_model` with adaptive length.
- Run at least one inference path for `suffix-draft_model` with fixed per-method
  token counts.
- Confirm the modified runtime uses the installed package path rather than an
  unrelated source checkout.

### 3. Environment Validation

- Prefer local GPUs visible through `nvidia-smi`.
- Target a two-GPU `Qwen3-32B` run on RTX 5090 devices.
- Reuse model weights from `/data` and `/data/models` when available.
- Treat functional bring-up as the hard gate; collect observability signals for
  dynamic length selection, but do not require a performance target in this
  design cycle.

## Implementation Notes for the Next Phase

The implementation plan should explicitly separate:

1. Config-surface adaptation
2. Runtime proposer wiring
3. Adaptive draft-length metrics and invocation
4. Long-context safety carry-over
5. Local validation on the installed package path

That decomposition will make it easier to execute and verify changes without
mixing unrelated speculative-decoding concerns into one edit pass.
