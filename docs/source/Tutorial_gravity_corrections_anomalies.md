# Calculating gravity anomalies

gSolve calculates three types of gravity anomaly, Free Air Anomaly, Simple Bouguer Anomaly and Complete Bouguer Anomaly. gSolve anomalies calculate normal gravity on the ellipsoid surface.

gSolve gravity anomalies are calculated as follows.

## Free Air Anomaly

***FAA = AG -(NG + FAC)***

where:

- FAA = Free Air Anomaly

- AG = Absolute Gravity

- NG = Normal Gravity calculated on the ellipsoid surface

- FAC = Free Air Correction

## Simple Bouguer Anomaly

***SBA = AG - (NG + FAC + AC + BSC + SBC)***

Where:

- SBA = Simple Bouguer Anomaly

- AG = Absolute Gravity

- NG = Normal Gravity calculated on the ellipsoid surface

- FAC = Free Air Correction

- AC = Atmospheric Correction

- BSC = Bouguer Slab Correction

- SBC = Spherical Bouguer Cap Correction

The terrain correction is not included in this calculation.

## Complete Bouguer Anomaly

***CBA = AG - (NG + FAC + AC + BSC + SBC - TC)***

Where:

- SBA = Simple Bouguer Anomaly

- AG = Absolute Gravity

- NG = Normal Gravity calculated on the ellipsoid surface

- FAC = Free Air Correction

- AC = Atmospheric Correction

- BSC = Bouguer Slab Correction

- SBC = Spherical Bouguer Cap Correction

- TC = Terrain Correction.
