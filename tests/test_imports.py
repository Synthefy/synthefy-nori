def test_public_imports():
    import synthefy_nori

    assert synthefy_nori.__version__
    assert callable(synthefy_nori.predict)
    assert callable(synthefy_nori.infer)
    assert synthefy_nori.NoriRegressor is not None
    # The classifier is intentionally not part of the published package.
    assert not hasattr(synthefy_nori, "NoriClassifier")
