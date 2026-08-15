# Release PR 404 dataframe forecasting through the public client

Status: accepted for PR 404 release preparation

Implementation: https://github.com/Synthefy/synthefy-nori-internal/pull/404

## Decision

Release the DataFrame forecasting additions in the lightweight `synthefy`
distribution as version 7.0.1. Version 7.0.0 is already immutable on PyPI.

`NoriTSForecaster` owns time-series validation, feature generation, and forecast
reconstruction. It accepts a configured `SynthefyNoriClient`, and every prepared
series executes through that client's existing `predict` contract. Do not add a
second `predict_df` implementation to `SynthefyNoriClient`; the client continues
to own backend transport while the forecaster owns workflow orchestration.

The public package README and changelog must show `future_df=`,
`target_column=`, and injected-client usage. Contract coverage must instantiate
the real public client and prove that the DataFrame workflow reaches its
`predict` method. `target_column=` selects one target per call; reject sequences
with an explicit unsupported-multiple-targets error until the workflow owns a
deliberate multi-target orchestration contract.

## Release scope

This is an L1 lightweight-client release. The feature performs horizon and
covariate preparation locally and reuses the existing regression request and
response contract, so it requires no Baseten deployment, gateway binding,
billing, key-management, model-weight, or heavy-package change.

Promote the identical patch from internal to staging to public. Only the public
repository may create `synthefy-v7.0.1` or publish the package.
