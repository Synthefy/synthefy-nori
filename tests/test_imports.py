def test_public_imports():
    import synthefy_nori

    assert synthefy_nori.__version__
    assert callable(synthefy_nori.predict)
    assert callable(synthefy_nori.infer)
    assert synthefy_nori.NoriRegressor is not None
    # NoriClassifier is re-exposed for the RelBench classification tasks
    # (entity-table tabular protocol); it reuses the trained cls head shipped in
    # the checkpoint.
    assert synthefy_nori.NoriClassifier is not None
