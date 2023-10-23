Package and Environment Management System
=========================================

We recommend using either Miniconda or Mamba to install Python and create a new environment. 
In this tutorial, we will use Mamba, but the steps are the same for Miniconda.

.. note::
    **From the Mamba documentation:**
    Mamba is a drop-in replacement and uses the same commands and configuration options as Conda.
    You can swap almost all commands between Conda and Mamba:

    .. code-block:: console

        mamba install ...
        mamba create -n ... -c ... ...
        mamba list

    The only exception is the `conda activate` command, which is replaced by `mamba activate`.


To intall python using mamba you need the mambaforge installer. You can download it from the miniforge's
github page:

.. note::

    `Mambaforge's Github page <https://github.com/conda-forge/miniforge#mambaforge>`_

Then you scroll to the mambaforge section and download the installer for your operating system.
This will install the mambaforge package and environment management system with mamba installed
in the base environment as shown in this image:

.. figure:: /imgs/mambaforge_github_page.png
   :scale: 100%
   :align: center
   :alt: The section where the user will download the mambaforge installer from the miniforge's github page

   The section where the user will download the mambaforge installer.

   In this section you can observe that there are options for different operating systems, including linux, mac and windows. 
   The user should download the installer for the operating system they are using. 
   In this case, we will download the windows installer, `Mambaforge-Windows-x86_64`.

After the download is completed you can run the installer and follow the instructions. After the installation
is completed there will be a new program in your start menu called `Mambaforge`. You can open it and use the
mambaforge terminal to create a new environment with the required dependencies.