API description
===============

The current package includes a series of functions that facilitate cronobiology analysis. 

The package itself has 4 files:

    #. **chrono_plotter.py**: contains functions to plot data
    #. **chrono_reader.py**: contains functions to read data
    #. **chrono_rythms.py**: contains functions to 
    #. **chrono_simulation.py**: contains functions to create simulated data

.. note::
    You can find more information about the usage of each function at the documentation of each file in the
    :doc:`introductory tutorial section under the user guide<../user_guide/pipelines>`



Chrono Reader
-------------

The ``chrono_reader`` module provides functionalities for reading data from various sources and preparing it for analysis and visualization within the **CircadiPy** package. It offers the following features:

- **Data Source Support**: The module is capable of reading data from multiple sources, including the ER4000 receiver, the Intellicage system, and generic ``.asc`` files, provided they adhere to specific formatting patterns for each source.

- **Protocol Object Creation**: Upon reading and processing the data, the module generates a protocol object that encapsulates relevant information necessary for subsequent analysis and plotting tasks.

- **Workflow Optimization Functions**: The module offers utility functions that enhance the workflow when working with the protocol object. One example is a method for concatenating data from multiple experiments seamlessly.

- **Data Exploration Assistance**: The module includes methods designed to facilitate data exploration. These methods are particularly useful when experimenters need to extract data from specific time intervals of interest.

- **Filtering Capabilities**: A built-in method allows the application of filtering techniques such as the Savitzky-Golay filter and moving average filter to the data.

- **Normalization Methods**: The module provides a method to normalize the data using either a z-score transformation or the min-max scaling approach.

The ``chrono_reader`` module acts as an essential foundation for data input, preprocessing, and initial exploration within the **CircadiPy** package, enabling researchers to efficiently prepare their data for in-depth chronobiological analysis and visualization.
