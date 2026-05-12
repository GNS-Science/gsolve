.. title:: Home

.. grid::
    :gutter: 2 3 3 3
    :margin: 5 5 0 0
    :padding: 0 0 0 0

    .. grid-item::
        :columns: 12 8 8 8

        .. raw:: html

            <h1 class="display-1">GSolve</h1>

        .. div:: sd-fs-3

            Processing gravity data

    .. grid-item::
        :columns: 12 4 4 4

        .. image:: ./_static/gsolve_logo.png
            :width: 200px
            :class: sd-m-auto dark-light

**GSolve** is a Python library for processing gravity data.

It includes common processing steps, like drift, tide and network adjustment, as well as calculation of corrections for free air, simple and complete Bouguer anomalies, including terrain_corrections.

.. grid:: 1 2 1 2
    :margin: 5 5 0 0
    :padding: 0 0 0 0
    :gutter: 4

    .. grid-item-card:: :octicon:`rocket` Getting started
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        New to GSolve? Start here!

        .. button-ref:: installation
            :click-parent:
            :color: primary
            :outline:
            :expand:

    .. grid-item-card:: :octicon:`book` GSolve theory
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        Gsolve Algorithms

        .. button-ref::  gsolve_algorithms
            :click-parent:
            :color: primary
            :outline:
            :expand:

             Algorithms for processing gravity data

    .. grid-item-card:: :octicon:`beaker` Tutorials
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        Tutorials

        .. button-ref::  Tutorial_network_adjustment_and_anomaly_calculation
            :click-parent:
            :color: primary
            :outline:
            :expand:

            Examples of how to use GSolve




    .. grid-item-card:: :octicon:`code` Reference documentation
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        A list of modules and functions

        .. button-ref:: api/api_index
            :click-parent:
            :color: primary
            :outline:
            :expand:


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Getting started
    :includehidden:

    installation.md
    fundamentals.md


.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Tutorials
    :includehidden:

    Tutorial_network_adjustment_and_anomaly_calculation
    Tutorial_calculate_calibration_beta_factor
    Tutorial_terrain_correction
    Tutorial_gravity_corrections_anomalies
    Tutorial_CG6_data

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Algorithms
    :includehidden:

    gsolve_algorithms
    gsolve_cli_manual
    terrain_corrections

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Reference documentation
    :includehidden:

    api/api_index.rst
    history
    authors
    development

