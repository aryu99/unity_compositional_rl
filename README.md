# unity_compositional_rl_agents

## Installation

Use Anaconda (https://www.anaconda.com/products/individual) to create a virtual environment with the appropriate packages.
After downloading and installing Anaconda 3, edit environment.yml so that the proper cuda toolkit is being installed (the file is currently configured for CUDA 11.3). 
To create the virtual environment, run:

> conda env create -f environment.yml

Gurobi optimization software is also necessary. This can be downloaded from [https://www.gurobi.com/downloads/](https://www.gurobi.com/downloads/).
Academic Gurobi licenses may be requested from [https://www.gurobi.com/downloads/end-user-license-agreement-academic/](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).

## Running the example

To run the example, change the current directory to the examples folder and run:
> python run_unity_labyrinth.py
