Environment Creation
--------------------

We provide an environment file called `circadipy_env.yml` that contains all the dependencies required to run
circadipy. To create a new environment using this file, you will need to use the newly installed mambaforge terminal.
To create the environment, you can follow these steps:

#. Open the mambaforge terminal.

#. Navigate to the folder where you downloaded the `circadipy_env.yml` file:

    * You can use the `cd` command to navigate to the folder.

    * E.g.: 
    
        .. code-block:: console 

           (base) $ cd "C:\Users\user\Downloads"

#. Run the following command:

    .. code-block:: console

       (base) $ mamba env create -f circadipy_env.yml

#. Wait for the environment to be created:

    * This can take a few minutes.

    * In the case of conda, the environment creation can take a significant amount of time because of the environment solving process. This process is usually faster with mamba.

#. Activate the environment using the following command:

    .. code-block:: console

       (base) $ mamba activate circadipy_env

#. Then the terminal should look something like this:

    .. code-block:: console

       (circadipy_env) $

