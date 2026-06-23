"""Tests for the published pricing API: ``synthefy_nori.pricing.billable_price``.

``billable_price`` is the only pricing symbol shipped in the installed package.
"""


def test_billable_price_is_on_the_public_api():
    # Importable from the top-level package and from the submodule (same
    # object), and advertised in __all__.
    import synthefy_nori
    from synthefy_nori import billable_price as top_level
    from synthefy_nori.pricing import billable_price as submodule

    assert top_level is submodule
    assert "billable_price" in synthefy_nori.__all__


def test_pricing_module_exposes_only_billable_price():
    # The shipped module carries just billable_price — not usage,
    # compute_tokens, or compute_price.
    from synthefy_nori import pricing

    public = {n for n in vars(pricing) if not n.startswith("_") and callable(getattr(pricing, n))}
    assert public == {"billable_price"}


def test_packaged_price_is_a_plain_float():
    # Returned over JSON, so it must be a builtin float, not a numpy scalar.
    from synthefy_nori.pricing import billable_price

    assert type(billable_price(1000, 10)) is float
