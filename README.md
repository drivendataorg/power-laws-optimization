 # Power Laws: Optimizing Demand-side Strategies

 ### For more about this repository, see the competition page:
 ### https://www.drivendata.org/competitions/53/optimize-photovoltaic-battery/

This repository contains the example implementation for how the optimization challenge code will be executed at the end of the competition. Competitors are required to only implement a single method `propose_state` within the file `battery_controller.py`.

This code exists to make it easy for competitors to test their solutions.

## Requirements
 - Docker
 - Python 3.6  (for local execution instead of on Docker)

## Running the simulation
 1. Clone this repository
 2. Add the data from the competition to the `data` folder
 3. Copy your implementation of `battery_controller.py` into `simulate/battery_controller.py`
 4. Execute the run script: `./run.sh`
 5. Your results will be stored and timestamped in the `all_results` folder each time you execute `run.sh` (Note: `results.csv` in `output` will be overwritten on each subsequent run.)

 **Note for Windows Users: We will accept a pull request with a `run.bat` script that works on Windows machines.**

The only supported execution is within Docker. However, if you want to run the Python code locally rather than in a Docker container, you can still use the `entrypoint.sh` script on unix based system. You may need to install `coreutils` on your system in order to use the `timeout` command. For OSX you can run `brew install coreutils` and then update `entrypoint.sh` to call `gtimeout` instead of `timeout`. Windows users will need to create their own script, although for basic purposes just running `python simulate/simulate.py` should be sufficient.

File | Description
---- | -----
`├── data` | A directory that has all of the input data as `.csv`s that are provided by the competition.
`├── output` | A directory for storing the output of a single simulation run.
`├── all_results` | This directory contains results from all of the runs executed.
`├── simulate` | The Python code for the simulation.
`|   ├── __init__.py` |
`|   ├── battery.py` | Contains an object for storing information about the battery.
`|   ├── battery_controller.py` | Implemented by competitors
`|   └── simulate.py` | Main entrypoint. Controls and executes the simulations.
`├── Dockerfile` | The definition for the Docker container on which the simulation executes.
`├── README.md` | About the project.
`├── entrypoint.sh` | Called inside the container to execute the simulation. Can also be used locally.
`├── requirements.txt` | The Python libraries that will be installed. Only the libraries in this official repo will be available.
`└── run.sh` | The only command you need. Builds and runs simulations in the Docker container.
