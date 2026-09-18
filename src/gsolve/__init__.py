from __future__ import annotations  # ruff: ignore[undocumented-public-package]

import importlib.metadata

try:  # ruff: ignore[non-empty-init-module]
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "5.X.X.dev"

from gsolve.meter_conversion import LaCosteRombergDialConverter
from gsolve.observations import GravityObservations, GravitySurvey
from gsolve.reductions.anomalies import GravityAnomalies
from gsolve.reductions.corrections import (
    GravityCorrectionParameters,
    GravityCorrectionProvider,
)
from gsolve.reductions.terrain_corrections import (
    TerrainCorrectionData,
    TerrainCorrectionParameters,
    TerrainCorrector,
)
from gsolve.reports import GSolveReport
from gsolve.sites import GravitySites, ReferenceGravity

__all__ = [
    "GSolveReport",
    "GravityAnomalies",
    "GravityCorrectionParameters",
    "GravityCorrectionProvider",
    "GravityObservations",
    "GravitySites",
    "GravitySurvey",
    "LaCosteRombergDialConverter",
    "ReferenceGravity",
    "TerrainCorrectionData",
    "TerrainCorrectionParameters",
    "TerrainCorrector",
]
