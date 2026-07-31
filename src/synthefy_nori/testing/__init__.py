"""Shared fixtures for testing the serving core, importable by every runner.

Not test code and not a runtime dependency of the engine — this is the *data* that lets two
very different runners assert the same thing:

- ``tests/test_memory_policy_e2e.py`` drives the engine in-process on a local GPU
- the live-deployment smoke suite POSTs the same table to a real Baseten deployment
  (scoped to #312, so today the e2e test is the only in-repo consumer)

It lives inside the package (rather than under ``ci/`` or ``tests/``) so both import it by the
one root they already use — ``serving/`` locally, ``/packages`` in a container — instead of each
inserting its own ``sys.path`` entry. The same convention the model package uses, where
``src/synthefy_nori/inference/test_memory_policy.py`` sits beside the code it tests.

**Stdlib only, and it must stay that way.** The deploy-time smoke runs on a bare ``python``
with no ``uv sync``, so nothing here may import numpy, torch or pydantic.

That matters for the deploy-time smoke, which runs on a bare ``python``: it must load
``rung_cases`` **by file path** rather than with a plain ``import synthefy_nori.testing``,
because a normal import executes ``synthefy_nori/__init__`` and pulls torch. If you write
that smoke (#312), the path-loading is load-bearing — do not "simplify" it into an import.

(When this module lived at ``nori_serving.testing`` an ordinary import *was* safe, because
that package's ``__init__`` is stdlib-only. Moving it into the model package inverted the
argument, and this docstring asserted the old one for a while.)

It ships in the PyPI wheel (setuptools finds every package under ``src/``). A few KB of stdlib
Python, and being installable is the whole point -- that is what makes it reachable from another
repo.
"""
