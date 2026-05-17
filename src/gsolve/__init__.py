from __future__ import annotations

import importlib.metadata

try:
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
    "GravityObservations",
    "GravitySurvey",
    "GravitySites",
    "ReferenceGravity",
    "LaCosteRombergDialConverter",
    "GSolveReport",
    "GravityCorrectionProvider",
    "GravityCorrectionParameters",
    "GravityAnomalies",
    "TerrainCorrectionParameters",
    "TerrainCorrector",
    "TerrainCorrectionData",
]
