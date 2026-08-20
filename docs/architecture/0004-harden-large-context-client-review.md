# Harden the large-context client/serving review findings

Status: accepted for the internal integration branch

Issue: https://github.com/Synthefy/nori-monorepo/issues/216

Implementation: https://github.com/Synthefy/synthefy-nori-internal/pull/447

## Decision

Apply the `/code-review` findings against PR 447 before it is merge-ready.
Keep the shared client/wire contract, the bounded hosted policy menu, and the
one-shot request semantics from ADR 0003 unchanged.

`SynthefyNoriClient.predict` now rejects `large_context_threshold` or
`large_context_seed` passed without a policy instead of silently dropping
them, using `None` (not the resolved default value) as the "omitted" sentinel
so an explicit value that happens to equal the default is still caught. The
three large-context fields are now passed into `NoriPredictRequest`'s
constructor together rather than assigned onto the built request afterward,
so the model's "threshold/seed require a policy" validator no longer depends
on an assignment-order comment. The local-mode capability probe for
`large_context_policy=` switched from a constructor-signature check to the
same `find_spec("synthefy_nori.inference.large_context")` module-presence
check `_local_memory_policy_available` already uses, and the corresponding
attribute assignment on the cached local regressor is now guarded against
`AttributeError`, not just `hasattr`. The cached local regressor's
`memory_policy`/`large_context_*` mutation and its fit/predict/report
sequence run under a new per-client lock, mirroring the shared serving
engine's own lock. A non-finite `y_train` now fails the remote HTTP transport
with a clear, actionable error instead of an unrelated stdlib `ValueError`
escaping from inside the retry loop.

On the serving side, the engine now fails closed with a 500 when a policy is
requested but the installed `synthefy-nori` does not implement it (checked
the same module-presence way as the client, not a regressor constructor
signature, so a test double or a future `**kwargs`-forwarding regressor is not
misclassified as unsupported), rather than silently reporting a plausible but
wrong `applied=False` capability handshake. The two Thinking-incompatible
capability checks (`output_type`, `large_context_policy`) share one helper
instead of two copy-pasted blocks. The Snowflake SPCS row-arity error now
names the actual row length and only blames `large_context_policy` for the
one row length that plausibly implies it.

Human review of the PR raised two further points, addressed in the same
branch. First, `large_context_threshold` had no validation at all on the
local `NoriRegressor` path: `large_context_applies` only ever compares
`n_train > threshold`, so a non-positive threshold silently made the policy
apply to every table regardless of size. `fit()` now rejects a non-positive
threshold the same way it already rejects an unresolvable policy name --
no upper bound is added locally, since `MAX_LARGE_CONTEXT_THRESHOLD` is a
network sanity/DoS cap for the untrusted, multi-tenant hosted contract, and a
local call has neither trust boundary. Second, the three hosted-safe
policies (`random`, `cluster_route`, `cluster_route_g4`) had no comparative
documentation anywhere a caller would see it before choosing one. All three
docstrings that describe the policy menu -- `NoriRegressor.__init__`,
`SynthefyNoriClient.predict`, and the hosted `NoriPredictRequest` schema
(both the client docstring and the server's OpenAPI field description) --
now state each policy's call count, its coverage on the validated benchmark
sweep, and which one is recommended (`cluster_route`, not `cluster_route_g4`
-- see ADR 0003's sweep numbers).

## Validation

A new parametrized test asserts the client's `_validate_large_context_controls`
and the server's `_parse_large_context` agree on the same bounds/coherence
edge cases, guarding the two independent hand-written implementations against
drift. The whole-function-skip regression in
`test_sdk_models_cover_every_customer_valid_case` is fixed to exclude only the
cases the pinned released client cannot represent, instead of skipping every
other case's field-preservation check too. The released-lane OpenAPI/client
schema-parity test now allows only this feature's own known additive fields to
differ, rather than skipping the bidirectional check for the whole lane. The
provenance manifest records the reviewed source-byte transitions for
`nori_client.py` and `nori_data_models.py`.

## Deferred and unchanged

No change to the bounded hosted policy menu, the billing/metering gate, or
the Snowflake SPCS protocol boundary from ADR 0003. Live serving validation
and public promotion remain gated until the internal branch passes its
separate release checks.
