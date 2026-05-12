# Calculate the terrain correction

```{tip}  Example script to calculate the terrain correction using the terrain_correction_example script in the examples\scripts folder.
```

Import some modules

```python
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gsolve import (
    GravitySites,
    TerrainCorrectionParameters,
    TerrainCorrector,
)
```

Setup paths to directories and files.

```python
data_path = pathlib.Path(__file__).parent.parent

grid_dir = data_path / "surveys" / "terrain_corrections"

data_file = "RIG_Terrain160_2610.xlsx"
dem_file1 = grid_dir / "Wellington8m_Land.tif"
dem_file2 = grid_dir / "wellington200m_Land.tif"
```

## load file with station locations and elevations

This can be done using the GravitySites.from_excel() method

```python
sites = GravitySites.from_excel(
    grid_dir / data_file,
    sheet_name="Locations",
)
```

## Setup the terrain correction zones

For each distance zone, usually associated with different DEM resolutions, set up the zone parameters.  Note only clamp_elevation for the inner most zone, or zone with the most detailed DEM.

```python
inner_zone_params = TerrainCorrectionParameters(
    min_dist=160.0,  # in meters
    max_dist=2160.0,  # in meters
    terrain_density=2670.0,  # in kg/m3
    water_density=1030.0,  # in kg/m3
    distance_mask_type="radial",
    dem_source=grid_dir / dem_file1,
    clamp_elevation=True,
    name="8m_dem",
)

outer_zone_params = TerrainCorrectionParameters(
    min_dist=2160.0,  # in meters
    max_dist=21900.0,  # in meters
    dem_source=grid_dir / dem_file2,
    name="200m_dem",
    clamp_elevation=False,
)
```

## Create a terrain corrector object

```python
corrector = TerrainCorrector(params=[inner_zone_params, outer_zone_params])
```

## Compute the terrain correction

```python
results = corrector.compute(
    points=sites,
    show_progress=True,
)
```

If you have tqdm installed a progress bar will show ```show_progress=True```

## Save the results

```python
results.to_excel(
    grid_dir / "terrain_correction_results.xlsx",
    sheet_name="Terrain_Correction",
    if_workbook_exists="replace",
)
```

Once terrain corrections have been calculated the results can be read back in during a standard work flow.

```python
from gsolve.reductions.terrain_corrections import TerrainCorrectionOutput

results_reread_from_disk = TerrainCorrectionData.from_excel(
  grid_dir / "terrain_correction_results.xlsx"
)
```

This will make it available for inclusion in complete Bouguer anomaly calculations.
