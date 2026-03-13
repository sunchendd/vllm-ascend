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
   `num_speculative_tokens_per_method: dict[str, int]` for
   `suffix-draft_model`, with exactly `suffix` and `draft_model` entries.
3. Pure `draft_model` must optionally support an adaptive draft-length range via
   `speculative_token_range`.
4. `suffix-draft_model` must remain fixed-length and must not consume
   `speculative_token_range`.
5. For pure `draft_model`, top-level `num_speculative_tokens` remains required
   and acts as the initial adaptive length.
6. For `suffix-draft_model`, top-level `num_speculative_tokens` is derived
   internally as `max(num_speculative_tokens_per_method.values())` for capacity
   planning and is not an additional required user input.
7. The runtime must instantiate and invoke both suffix and draft proposers for
   `suffix-draft_model`, with suffix results preferred and draft-model fallback
   used when suffix speculation does not supply tokens.
8. The pure `draft_model` path must continue to use a single
   `DraftModelProposer`, with the selected draft length computed dynamically
   from recent acceptance behavior.
9. The modified package must work from the installed path under
   `/usr/local/lib/python3.12/dist-packages/vllm`.

### Non-Functional Requirements

1. Prioritize safe functional adaptation over full optimization parity with all
   historical patches.
2. Avoid introducing interface ambiguity between fixed-length
   `suffix-draft_model` and adaptive-length pure draft mode.
3. Preserve compatibility with the existing `0.17.0` proposer and scheduler
   structure.
4. Keep validation centered on local bring-up and successful inference rather
   than performance targets.

## Scope

### In Scope

- Explicit `suffix-draft_model` configuration support.
- Per-method speculative token-count validation for `suffix-draft_model`.
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
  `suffix-draft_model` is routed through the correct runtime branches.
- Keep the public ingress on the existing path only:
  `vllm/engine/arg_utils.py:create_speculative_config()` continues to build
  `SpeculativeConfig` from CLI JSON or engine-supplied dicts, and the new fields
  are added to that existing payload rather than introducing a second parser.
- Enforce hard validation rules:
  - `suffix-draft_model` requires `num_speculative_tokens_per_method`.
  - `suffix-draft_model` requires both `suffix` and `draft_model` entries.
  - `suffix-draft_model` rejects extra keys in
    `num_speculative_tokens_per_method`.
  - `suffix-draft_model` rejects `speculative_token_range`.
  - `suffix-draft_model` rejects any user-supplied top-level
    `num_speculative_tokens`; the runtime derives the capacity value internally
    from `num_speculative_tokens_per_method`.
  - pure `draft_model` requires top-level `num_speculative_tokens`.
  - Pure `draft_model` may use `speculative_token_range`, but only with valid
    positive integer values.

The configuration layer remains the single point that defines whether the
execution path is fixed-length `suffix-draft_model` or adaptive-length pure
draft mode.

### 2. Runtime Routing Layer

File focus: `vllm/v1/worker/gpu_model_runner.py`

Responsibilities:

- Keep the existing pure `draft_model` path intact, still backed by one
  `DraftModelProposer`.
- Add combination-mode routing for `suffix-draft_model`, where:
  - a suffix proposer is instantiated for suffix speculation;
  - a draft-model proposer is instantiated for fallback speculation;
  - `GPUModelRunner` creates two immutable derived `VllmConfig` copies during
    drafter setup:
    - `suffix_vllm_config` binds `method='suffix'` and
      `num_speculative_tokens = num_speculative_tokens_per_method['suffix']`;
    - `draft_vllm_config` binds `method='draft_model'` and
      `num_speculative_tokens = num_speculative_tokens_per_method['draft_model']`;
  - no new suffix-proposer API is required for token-count control; method-
    specific token counts are supplied through those derived config copies;
  - suffix output is attempted first;
  - the routing contract is per request and mutually exclusive for a decode
    step: if suffix speculation returns one or more tokens for a request, that
    request uses the suffix result as-is for that step and is not topped up by
    draft-model output;
  - draft-model speculation is invoked only for requests whose suffix proposer
    returns an empty token list for that step.
- `GPUModelRunner.propose_draft_token_ids()` owns the merge boundary for this
  mode and keeps the existing full-batch container shape:
  - it first calls the suffix proposer on the full current batch, using the same
    `input_batch.req_ids` order as `0.17.0`;
  - the suffix proposer returns a full-batch `list[list[int]]`, with one entry
    per request;
  - if every request receives a non-empty suffix proposal, that full-batch list
    is returned directly;
  - otherwise `GPUModelRunner` computes an ordered `suffix_miss_indices` list
    over `input_batch.req_ids` for requests whose suffix proposal is empty;
  - `GPUModelRunner` then builds a reduced fallback batch containing only
    `suffix_miss_indices`, by projecting every per-request proposer input
    (`req_ids`, sampled-token container, request objects, and the matching
    rows/spans of attention metadata and slot mappings) into the same relative
    order as `suffix_miss_indices`;
  - the draft-model proposer runs only on that reduced fallback batch using
    `draft_vllm_config`, so suffix-hit requests produce no draft-side KV or
    runtime state effects;
  - the fallback result is merged back by writing
    `draft_fallback_ids[k]` into the original batch slot
    `suffix_miss_indices[k]`;
  - the merged result is a full-batch `list[list[int]]` aligned to
    `input_batch.req_ids`, so `_calc_spec_decode_metadata()` continues to derive
    `num_draft_tokens` by taking `len(draft_token_ids)` per request with no new
    verifier-side container.
- The downstream verifier interface does not change. It still receives one
  per-request speculative proposal for the current decode step.
- Track speculative-decoding acceptance statistics for pure `draft_model`.
- Compute and pass the next draft length into the draft proposer only for pure
  `draft_model`.
- Keep `suffix-draft_model` on fixed token counts derived from
  `num_speculative_tokens_per_method`.
- Own the adaptive controller instance for pure `draft_model`; the scheduler
  continues to emit aggregated speculative stats, while the runner alone decides
  when to update and apply the next draft length.

This preserves a clean separation:

- `draft_model` => adaptive single-proposer path
- `suffix-draft_model` => fixed-length dual-proposer path

### 3. Draft Proposer Layer

File focus: `vllm/v1/spec_decode/draft_model.py`

Responsibilities:

- Extend `DraftModelProposer.propose()` and only the internal helpers it calls
  directly to accept an explicit `draft_length: int | None = None`.
- When `draft_length` is `None`, the proposer preserves current `0.17.0`
  behavior by using `self.num_speculative_tokens`.
- Use the provided draft length in place of always consuming the configured
  maximum speculative-token count.
- Keep confidence-based early-stop handling internal to the proposer.
- Ensure stopped sequences do not corrupt KV state by applying the appropriate
  mask and slot-mapping behavior.
- Preserve compatibility with existing draft-model loading and model-tagging
  behavior in `0.17.0`.

Caller blast radius is intentionally narrow:

- Pure `draft_model` calls in `GPUModelRunner.propose_draft_token_ids()` pass
  the adaptive `draft_length`.
- `suffix-draft_model` fallback calls in `GPUModelRunner` pass the fixed
  configured draft-model token count or allow the proposer default to resolve to
  the same fixed value.
- The only expected signature-alignment file outside `draft_model.py` is
  `vllm/v1/spec_decode/eagle.py`, because `SpecDecodeBaseProposer.propose()` is
  defined there.

### 4. Metrics Layer

File focus: `vllm/v1/spec_decode/metrics.py`

Responsibilities:

- Record aggregate acceptance behavior for pure `draft_model`.
- Maintain a smoothed acceptance signal suitable for selecting the next draft
  length from `speculative_token_range`.
- Expose a deterministic helper for computing the next draft length.

Adaptive-length state ownership is engine-local rather than per request:

- `vllm/v1/spec_decode/metrics.py` owns the pure math and state-transition rules
  for adaptive draft length.
- `GPUModelRunner` owns one adaptive draft-length controller instance for its
  local pure-`draft_model` execution path and is the only runtime component that
  mutates or reads that controller during decode.
- The scheduler does not own adaptive policy state. It only emits the existing
  per-iteration aggregate `SpecDecodingStats` through
  `Scheduler.make_spec_decoding_stats()` and publishes it as
  `SchedulerStats.spec_decoding_stats` in `vllm/v1/metrics/stats.py`.
- The implementation plan must include one explicit plumbing hook that transfers
  `SchedulerStats.spec_decoding_stats` into the runner-owned adaptive controller
  before the next pure-`draft_model` proposal step.
- Pure `draft_model` keeps `num_speculative_tokens` as a required startup field
  even when adaptive length is enabled. `speculative_token_range` is an
  additional control surface, not a replacement for the existing startup token
  count.
- Startup validation requires `num_speculative_tokens` to appear in
  `speculative_token_range`; otherwise initialization fails explicitly. The
  initial adaptive draft length is exactly `num_speculative_tokens`.
- The controller updates once per decode iteration from the batch-aggregated
  speculative acceptance stats already produced by the scheduler. The scheduler
  continues to emit one `SpecDecodingStats` object per decode iteration, and the
  adaptive controller consumes only `num_drafts`, `num_draft_tokens`, and
  `num_accepted_tokens` from that object.
- Update timing is explicit:
  - verification for decode step `t` finishes first and produces the aggregate
    `SpecDecodingStats` for step `t`;
  - `GPUModelRunner` then updates the adaptive controller from those stats;
  - the resulting selected length is stored as `next_draft_length`;
  - that `next_draft_length` is applied on the next pure-`draft_model` proposal
    call for decode step `t + 1`, never retroactively to step `t`.
- If a decode iteration produces `num_draft_tokens == 0`, the adaptive
  controller leaves the current draft length unchanged.
- Smoothing uses an engine-local EWMA with `alpha = 0.2`:
  `acceptance_ewma = 0.8 * acceptance_ewma + 0.2 * batch_acceptance_rate`,
  where `batch_acceptance_rate = num_accepted_tokens / num_draft_tokens`.
- The EWMA starts at `0.6` on engine initialization so the runtime begins from a
  neutral, mid-acceptance assumption instead of immediately biasing upward or
  downward.
- Draft-length adjustment is single-step and deterministic over the sorted
  configured range:
  - if `acceptance_ewma >= 0.75`, move up by one configured length;
  - if `acceptance_ewma <= 0.45`, move down by one configured length;
  - otherwise, keep the current length.
- The selected draft length is always clamped to the configured
  `speculative_token_range`.
- The controller resets only when the engine or speculative configuration is
  rebuilt; it does not maintain per-request adaptation state.

Combination modes do not consume this adaptive control. Their token counts stay
fixed by configuration.

### 5. Graph Dispatch Layer

File focus: `vllm/v1/cudagraph_dispatcher.py`

Responsibilities:

- Prevent dynamic-length draft execution from being routed into incompatible
  uniform-decode cudagraph assumptions.
- Apply a concrete routing rule:
  - pure `draft_model` with `speculative_token_range` is excluded from the
    uniform full-graph path because draft width can change between decode steps;
  - `suffix-draft_model` is excluded from the uniform full-graph path because
    suffix-hit and suffix-miss requests produce mixed proposal widths in the
    same step;
  - in `CudagraphDispatcher.dispatch(...)`, these modes must be routed as
    `uniform_decode=False` or with `FULL` removed from `allowed_modes`, so only
    `PIECEWISE` or `NONE` can be selected;
  - on the proposer side, existing `SpecDecodeBaseProposer.initialize_cudagraph_keys()`
    behavior remains the ceiling: proposer execution may use `PIECEWISE` or
    `NONE`, but not `FULL`;
  - if no compatible cudagraph key exists, dispatch falls back to
    `CUDAGraphMode.NONE` rather than forcing uniform graph capture.

## Detailed Data Flow

### Pure `draft_model` with Adaptive Length

1. User config selects `method="draft_model"` and may provide
   `speculative_token_range`.
2. `SpeculativeConfig` requires `num_speculative_tokens` and validates that
   `speculative_token_range`, when present, is a strictly increasing list of
   unique positive integers that includes `num_speculative_tokens`.
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
   proposer from immutable derived `VllmConfig` copies created during drafter
   setup, so:
   - the suffix proposer sees
     `num_speculative_tokens = num_speculative_tokens_per_method['suffix']`;
   - the draft proposer sees
     `num_speculative_tokens = num_speculative_tokens_per_method['draft_model']`.
4. At proposal time, suffix speculation runs first across the full batch and
   returns `list[list[int]]` aligned to `input_batch.req_ids`.
5. If every request receives a non-empty suffix proposal, the runtime returns
   that full-batch result directly.
6. Otherwise the runtime computes `suffix_miss_indices` in original request
   order and projects a reduced draft fallback batch containing only those
   requests.
7. The draft proposer runs only on that reduced batch, so suffix-hit requests do
   not incur draft-side KV or runtime state effects.
8. The runtime writes each reduced-batch fallback result back to the original
   request slot indexed by `suffix_miss_indices[k]`.
9. The runtime does not concatenate suffix output with draft-model output in a
   single request for a single step.
10. The merged output remains a full-batch `list[list[int]]` in original request
    order, so existing downstream metadata calculation remains valid.
11. No adaptive draft-length selection is performed in this mode.

## Error Handling and Safety Rules

1. Invalid `suffix-draft_model` configurations must fail at config-validation
   time with explicit error messages.
2. Mixed fixed-length and adaptive-length controls must be rejected rather than
   silently ignored.
3. `speculative_token_range` must contain only positive integers, must not be
   empty, must be strictly increasing as supplied, and must not contain
   duplicates. Invalid ranges are rejected; they are not normalized or
   reordered.
   A single-value range equal to `num_speculative_tokens` is valid and acts as a
   fixed-length no-adaptation configuration.
4. `num_speculative_tokens_per_method` must contain only positive integer
   values. For `suffix-draft_model`, only the active method keys `suffix` and
   `draft_model` are accepted; extra keys are rejected to avoid ambiguous
   intent.
5. No legacy compatibility alias is planned for adaptive draft length in this
   cycle. The supported field name is `speculative_token_range`; older field
   names from historical patches are out of scope and should fail clearly if
   surfaced.
6. If suffix decoding is requested but `arctic_inference` is unavailable, the
   runtime must fail fast at initialization with an explicit dependency error.
   This is already aligned with the installed `SuffixDecodingProposer`, which
   lazily imports `arctic_inference.suffix_decoding`.
   Handling this dependency is in scope for implementation as an environment
   prerequisite, not as a change to speculative-decoding semantics.
7. Confidence-gated early stopping must remain an internal optimization; it must
   not require new user-visible control flow outside the speculative config.
8. Sequence-length, slot-mapping, and KV-write behavior must remain correct for
   partially stopped draft batches.
9. The long-context draft-model position safety fix must be carried over from
   local commit `9ccfe8206d8de8a96ebbfc2b9457274cfc16b618`, which references
   patch
   `/home/scd/vllm-ascend/vLLM-ascend/patch/0011-xfusion-vLLM-ascend-fix-draft-model-position-oob.patch`.
   In `0.17.0`, the owning implementation unit is the shared
   `SpecDecodeBaseProposer.propose()` position-clamping and slot-mapping block
   in `vllm/v1/spec_decode/eagle.py`, which is used by `DraftModelProposer`.
   The required behavior is to clamp draft-model positions against the draft
   model's own `max_position_embeddings` rather than the target model's
   `max_model_len`, while keeping sequence-length accounting on the original
   decode progression. Out-of-range draft positions must map to safe padded
   slots so verification rejects bad draft tokens without crashing the service.

## Planned File Touch Points

Primary files expected to change:

- `vllm/config/speculative.py`
- `vllm/engine/arg_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/spec_decode/draft_model.py`
- `vllm/v1/spec_decode/metrics.py`
- `vllm/v1/cudagraph_dispatcher.py`
- `vllm/v1/spec_decode/eagle.py`

Secondary files that may need change only if exact `0.17.0` helper signatures
force alignment:

- none expected beyond the primary file set

## Validation Strategy

### 1. Static and Startup Validation

- Confirm the modified installed package imports successfully.
- Confirm `SpeculativeConfig` parses:
  - pure `draft_model` with `speculative_token_range`
  - `suffix-draft_model` with `num_speculative_tokens_per_method`
- Confirm pure `draft_model` rejects startup when `num_speculative_tokens` is
  absent or not present in `speculative_token_range`.
- Confirm proposer initialization selects the intended runtime objects.

### 2. Functional Validation

- Run at least one inference path for baseline pure `draft_model` without
  `speculative_token_range` and confirm existing fixed-length upstream behavior
  is preserved.
- Run at least one inference path for pure `draft_model` with adaptive length.
- Run at least one inference path for `suffix-draft_model` with fixed per-method
  token counts.
- Confirm the modified runtime uses the installed package path rather than an
  unrelated source checkout.
- Confirm adaptive draft length changes at runtime by observing at least one
  decode session where the selected `draft_length` differs from its initial
  value and remains within `speculative_token_range`.
- Confirm the scheduler-to-metrics integration updates the adaptive controller
  only from the batch-level `SpecDecodingStats` fields named in this spec, so
  the implementation does not silently depend on per-request hidden state.
- Confirm `suffix-draft_model` fallback semantics explicitly:
  - requests with non-empty suffix speculation do not invoke draft-model top-up
    in that step;
  - requests with empty suffix speculation do invoke the fixed-length
    draft-model fallback in that step.

### 3. Negative and Boundary Validation

- Reject `suffix-draft_model` configs that omit either `suffix` or
  `draft_model` from `num_speculative_tokens_per_method`.
- Reject `suffix-draft_model` configs that include extra method keys in
  `num_speculative_tokens_per_method`.
- Reject `suffix-draft_model` configs that also provide user-supplied top-level
  `num_speculative_tokens`.
- Reject `suffix-draft_model` configs that also provide
  `speculative_token_range`.
- Reject malformed `speculative_token_range` values such as zero, negative,
  empty, duplicate, or non-monotonic ranges as supplied.
- Fail clearly when suffix decoding is requested without the required
  `arctic_inference` dependency.
- Exercise a long-context request where the target-model context exceeds the
  draft-model position-embedding limit and confirm the service remains up.
- Exercise a mixed batch where at least one request receives non-empty suffix
  speculation and at least one request receives empty suffix speculation, then
  confirm `GPUModelRunner.propose_draft_token_ids()` projects only the
  `suffix_miss_indices` requests into the draft fallback batch and returns one
  merged full-batch `list[list[int]]` aligned to `input_batch.req_ids` before
  verification.

### 4. Environment Validation

- Prefer local GPUs visible through `nvidia-smi`.
- Target a two-GPU `Qwen3-32B` run on RTX 5090 devices.
- Reuse model weights from `/data` and `/data/models` when available.
- Treat functional bring-up as the hard gate; collect observability signals for
  dynamic length selection, but do not require a performance target in this
  design cycle.
- If the target GPU/model setup is temporarily unavailable during a given run,
  fall back to config parsing and smaller local smoke validation for developer
  iteration, but do not treat that fallback as satisfying final acceptance.

## Implementation Notes for the Next Phase

The implementation plan should explicitly separate:

1. Config-surface adaptation
2. Runtime proposer wiring
3. Adaptive draft-length metrics and invocation
4. Long-context safety carry-over
5. Local validation on the installed package path

That decomposition will make it easier to execute and verify changes without
mixing unrelated speculative-decoding concerns into one edit pass.
