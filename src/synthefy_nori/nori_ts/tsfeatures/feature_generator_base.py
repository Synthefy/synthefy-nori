# Adapted from PriorLabs/tabpfn-time-series:
#   https://github.com/PriorLabs/tabpfn-time-series
#
# Copyright 2025 Prior Labs GmbH
# SPDX-License-Identifier: Apache-2.0
#
# Modifications by Synthefy: import paths rewritten for synthefy_nori. No TabPFN
# model code or weights are included — only the dependency-light time-feature
# engineering.

from abc import ABC, abstractmethod

import pandas as pd


class FeatureGenerator(ABC):
    """Abstract base class for feature generators"""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the given dataframe"""
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate(df)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.__dict__}"

    def __repr__(self) -> str:
        return self.__str__()
