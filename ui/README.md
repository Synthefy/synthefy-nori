# Nori Studio UI

An interactive public-data workspace for exploring Nori embeddings, explanations,
zero-shot inference, and scenario simulation.

## Run locally

```bash
npm install
npm run dev
```

The production build is created with `npm run build`.

## Demo data

`public/data/nori-embeddings.json` is the public, precomputed 3,000-customer
artifact used by the Synthefy website's Nori embeddings article. It is derived
from the UCI Default of Credit Card Clients dataset and contains target-aware
embedding projections, raw-feature projections, held-out labels, and three
display-safe customer attributes.

The embeddings screen renders that artifact directly. The explain screen uses
local calculations over its display-safe attributes to demonstrate the product
interaction and is labeled **Interface preview**. The zero-shot percentage is a
nearest-neighbor cohort baseline and is labeled **Static demo**. Neither is
presented as live Nori inference.

## Connecting live Nori

The production service can replace the local preview calculations with endpoints
backed by:

- `NoriRegressor.fit(...).predict(...)` for zero-shot inference
- `NoriRegressor.get_embeddings(...)` for target-aware embeddings
- `get_nori_imputation_explainer(...)` for Shapley values and SHAP-IQ interactions
- repeated predictions against a fixed context for controlled scenarios

Keep model execution server-side; the browser bundle should remain a thin,
responsive visualization layer.
