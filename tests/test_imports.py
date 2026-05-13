def test_public_imports():
    import synthefy_tabular

    assert synthefy_tabular.__version__
    assert callable(synthefy_tabular.predict)
    assert callable(synthefy_tabular.infer)
    assert synthefy_tabular.SynthefyTabularRegressor is not None
    assert synthefy_tabular.SynthefyTabularClassifier is not None
