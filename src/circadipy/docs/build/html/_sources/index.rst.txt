.. circadipy documentation master file, created by
   sphinx-quickstart on Tue Aug  1 12:37:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
Welcome to circadipy's documentation!
=======================================

Introducing **CircadiPy**, the a Python package for chronobiology analysis! 
With seamless integration of powerful time series plotting libraries, 
it empowers researchers to visualize and study circadian cycles with unrivaled versatility.

Currently, the package supports the visualization of biological rhythms and their synchronization with external cues using:

1. Actograms: An actogram is a graphical representation of an organism's activity or physiological data over time. It typically shows activity or physiological measurements (e.g., hormone levels, temperature) along the y-axis and time along the x-axis. Actograms are often used to visualize circadian rhythms and patterns of activity rest cycles.

2. Cosinor Analysis Plot: This plot is used to analyze and display the presence of rhythmic patterns in data. It's a graphical representation of the cosinor analysis, which fits a cosine curve to the data to estimate the rhythm's parameters like amplitude, acrophase (peak time), and period.

3. Raster Plot: A raster plot displays individual events or occurrences (such as action potentials in neurons) over time. In chronobiology, this can be used to show the timing of specific events in relation to the circadian cycle.

------------------------------------------------------------------------------------------------------------------------------

CircadiPy also provides a built-in generator of simulated data, making possible the creation of custom datasets for testing, experimentation and comparison purposes.

.. note::

   You can view the whole source code for the project on
   `Circadipy's Github page <https://github.com/nncufmg/circadipy>`_


.. toctree::
   :caption: FIRST STEPS
   :maxdepth: 2
   :hidden:

   /user_guide/first_steps

.. toctree::
   :caption: USER GUIDE
   :maxdepth: 2
   :hidden:

   user_guide/package_manager
   user_guide/env_creation
   user_guide/installation

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2
   :hidden:

   api_reference/chrono_plotter
   api_reference/chrono_reader
   api_reference/chrono_simulation
   api_reference/chrono_rithm