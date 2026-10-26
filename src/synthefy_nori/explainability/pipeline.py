"""Compatibility imports and CLI for the published explainability pipeline."""

from synthefy_nori.interpretability.glassbox.pipeline import SELECTION_FRACTION, main, run

__all__ = ["SELECTION_FRACTION", "main", "run"]

if __name__ == "__main__":
    main()
