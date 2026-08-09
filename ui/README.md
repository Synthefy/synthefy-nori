# Nori Studio UI

An interactive public-data workspace for exploring Nori embeddings, explanations,
zero-shot inference, and scenario simulation.

The opening dataset gallery offers five demos: the precomputed UCI credit-default
artifact plus bundled Palmer Penguins, Automobile MPG, Restaurant Tips, and
Titanic CSVs from the seaborn sample-data repository.

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

## Bring your own CSV

Use **Add dataset** to drop a local CSV or load a public, CORS-enabled CSV URL.
The browser reads up to 3,000 rows, detects numeric columns, and lets you choose
the outcome Nori should reason about. Local datasets get a working raw-feature
projection, correlation-based signal preview, nearest-neighbor baseline, and
interactive scenarios across the same four workspace lenses.

Target selection is required before a CSV can enter the studio. Nori Studio then
creates a deterministic seeded 80/20 context/test split. Classification targets
are stratified; continuous targets use a random split. Reference calculations use
only context rows, while selectable held-out rows are used as test queries across
embeddings, explanations, zero-shot inference, and scenarios.

Uploaded files stay in the current browser tab and are not sent to Nori Studio.
Linked files are fetched directly by the browser, so their host must allow CORS.
These local diagnostics are clearly labeled and are not represented as live Nori,
SHAP, or SHAP-IQ output.

## Connecting live Nori

The production service can replace the local preview calculations with endpoints
backed by:

- `NoriRegressor.fit(...).predict(...)` for zero-shot inference
- `NoriRegressor.get_embeddings(...)` for target-aware embeddings
- `get_nori_imputation_explainer(...)` for Shapley values and SHAP-IQ interactions
- repeated predictions against a fixed context for controlled scenarios

Keep model execution server-side; the browser bundle should remain a thin,
responsive visualization layer.
