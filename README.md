# nanoscale_measurement

A repository for code to aid in the measurement and analysis of nano-scale processes. Semiconductor fabrication, thin-film measurement, SEM and other microscopy techniques.

## Profilometry module

A module for analysis of 1-dimensional trace data from a profilometer instrument. Removal of trace artifacts, leveling of data, measurement of film thicknesses and display of data.

## INSTALLATION

- Install miniconda (or Anaconda), git and VSCode on your machine if not already running them.
- Setup Jupyter in VSCode.
- Clone this repo.
- Create conda environment to run the repo using 'nanoscale.yaml' config file:
*Note: execute commands, one line at a time.* Run these commands in your terminal shell, e.g powershell, cmd prompt (windows) or bash or zsh (linux & mac).

Set working directory to the cloned repo on your machine. Here "...." represents the path to this folder on your machine.

``` bash
cd ..../nanoscale_measurement
```

Create conda environement using the .yaml conifg. file. This will install the requirements for the repo.

``` bash
conda env create -f nanoscale.yaml
```

- You can then import the repo into a jupyter notebook (recommended method).
- Open the notebook: "Profilometry_run.ipynb". This notebook has a template of how to use the profilometry module in a jupyter notebook to measure a profilometer trace and plot data.
