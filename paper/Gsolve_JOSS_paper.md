---
title: 'GSolve - a python package for processing gravity data from relative terrestrial gravity meters'
tags:
  - Python
  - gravity
  - geophysics
  
authors:
  - name: Adrian Benson
    orcid: 0000-0003-1428-3016
    affilation: 1
  - name: Alison Kirkby
    orcid: 0000-0003-1361-440X
    affilation: 1
  - name: Craig Miller0000-0003-1361-440X
    orcid: 0000-0001-8499-0352
    affiliation: 1  
  - name: Alexsandr Spesitsev
    orcid:
    affilation: 1
  - name: Vaughan Stagpoole
    orcid: 0000-0002-2878-1238
    affilation: 1

affiliations:
 - name: Earth Sciences New Zealand
   index: 1
date: 20 July 2025
bibliography: paper.bib
---

# Summary
`GSolve` is an open-source software written in Python. The purpose of `GSolve` is to process raw data collected using terrestrial relative or absolute gravity meters for mapping and time-varying gravity surveys, through to products that are useful for geophysical interpretation.  This software is a complete rewrite of a previous Python 2 GUI-based version [@MCCUBBINE2018a] which itself was a rewrite of fortran code originally called DSolve [@WOODWARD1984].

# Statement of Need
Gravity data collected using relative or absolute terrestrial gravity meters provides valuable information on the density structure of the subsurface and are useful for many Earth science applications including mineral and geothermal exploration, volcanology, engineering geology and geological mapping.  Measurements of gravity change over time are useful for geothermal, oil and gas, hydrology and volcanology fields.  Processing data from terrestrial relative gravity meters such as Scintrex CG series instruments or LaCoste and Romberg G and D meters requires several steps to convert the relative gravity measurements to "reduced" data that can be analysed for its geoscientific purpose.  `GSolve` provides a modular system to undertake network adjustments and apply corrections including earth tide, ocean loading and terrain correction to calculate free-air, simple and complete Bouguer anomalies.
 
# State of the field
Several tools exist for gravity data processing through to the network adjusted solution step.  Prior to beginning the rewrite we undertook an extensive comparison study of available tools [@KIRKBY2023], summarised here.  

`Gtools` [@BATTAGLIA2022] is an open-source matlab package from the USGS designed for time-varying measurements that computes the drift-corrected network adjustment and includes ocean loading.  `GSAdjust` [@KENNEDY2020a] is a python tool, also from USGS, with a graphical user interface which computes the drift-corrected network adjustment.  `CG6TOOL`[REF]  is developed by the International Gravimetric Bureau and is the latest version of a series of software applications that supports Scintrex gravimeter data, beginning with CG3TOOL. It is written in Java and runs on both Linux and Windows platforms.  `CG6TOOL` [REF] carries out daily corrections including earth tides, drift by least-squares adjustment of repeated sites, station height adjustment using a free air gradient, and network adjustment.

Terrain correction computations (required for complete Bouguer anomlay) are typically only found in commercial software (e.g. Oasis Montaj or Intrepid), or are standalone programms in other free software, for example `TOPOSk` [@ZAHOREC2017a], requiring manual integration with other software to produce the final anomalies.

# Software design
`GSolve`'s design philosophy is based on the following core principles: (1) A modern object oriented, user-friendly API, that (2) meets opensource standards for documentation and testing.  We deliberately moved to an object-oriented API from the previous gui version and completely rewrote all functions.  

An important design decision is that where solutions to components of `GSolve` already exist, we chose to incorporate or warp them rather than rewrite our own.  For example, to implement the ETERNA tidal model we wrapped `pygtide`, whilst for providing ocean loading values we wrapped `pyhardisp`.  Normal gravity is calculated using Boule [@Boule] with a choice of ellipsoids available.  Likewise for terrain corrections we use the underlying functionality of `Harmonica` [@Harmonica] which efficiently computes the gravity response of a prism.

The software is built around a series of classes which XXXXX ADD PARAGRAPH ABOUT THE CLASS DESGIN STRUCTURE ETC.

# Research impact statement
From its original release in 1984 `Gsolve` has been used in numerous scientific papers and commercial reports.  The current version has been used in papers by [@MILLER2022], [@MILLER2025a], [@LUTHFIAN2023] etc.



# Features
`GSolve` is written in python 3 and provides API based functionality.  Pandas dataframes are utilised for input/output, storage and data manipulation, meaning that additional calculations not provided by GSolve (such as the differencing of time varying surveys) are easily possible.  

  * **Data import**: `GSolve` can read instrument files from Scintrex CG6 meters as well as from manually read or "Nomad" files from La Coste and Romberg G or D meters.  Non-standard files with appropriately labelled columns can be imported using read functions.  For manually read LaCoste and Romberg meters a meter calibration file is required to convert unitless dial mearurements to gravity in mGal.

  * **Network adjustment**: `GSolve` uses the algorithm of [@REILLY1970] for network adjustment with options for unconstrained, decoupled, or constrained least squares solutions using either a loop-based or "all-data" approach.  Reference or absolute gravity values used in the network adjustement are provided separately to the measured relative gravity measurements.

  * **Earth tide and ocean loading**: Earth tide corrections are applied using either Longman or ETERNA tide correction models [@Pygtide].  Ocean loading corrections can be computed using the `pyhardisp` wrapped functions or read from pre-computed results from other software e.g. Quick Tide Pro.

  * **Gravity corrections and anomalies**: `GSolve` calculates free-air, simple Bouguer and complete Bouguer anomalies. Free air, atmospheric, Bouguer slab, and spherical Bouguer slab corrections are computed using formula from [@HINZE2005].  Terrain corrections leverage Harmonica [@Harmonica] for rapid calculation of the topographic/bathymetric effect using a prism based approach.

  * **Data export**: Results tables for observations, sites, corrections and anomalies can be written to a multi-sheet excel spreadsheet where all meta-data for each processing step is included.  Output solutions include the adjusted gravity value with its standard deviation, standard error and variance, along with the drift rate.

  * **Visualisation**: Plots of network adjustment residuals as timeseries or CDF functions are provided along with drift curve plots.  A network map visualising loops can be made.

# Acknowledgements
`GSolve` development was supported by Earth Sciences New Zealand (previously GNS Science) capability development fund.

# AI usage disclosure.
Generative AI was used to write some unit tests and facilitate debugging.  All original code was written manually.

# References

