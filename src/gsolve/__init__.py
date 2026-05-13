from __future__ import annotations

import importlib.metadata as _im

try:
    __version__ = _im.version("gsolve")
except _im.PackageNotFoundError:
    __version__ = "5.6.0"

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
