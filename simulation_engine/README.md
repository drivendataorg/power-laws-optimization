<a href="https://www.drivendata.org/competitions/53/optimize-photovoltaic-battery/">
    <img src="https://s3.amazonaws.com/drivendata-public-assets/se-challenge-3-banner.jpg"/>
</a>

# Power Laws: Optimizing Demand-side Strategies

**For more about this code, see the [competition page](https://www.drivendata.org/competitions/53/optimize-photovoltaic-battery/).**

This directory contains the example implementation for how the optimization challenge code will be executed at the end of the competition. Competitors are required to only implement a single method propose_state within the file battery_controller.py.

This code exists to make it easy for competitors to test their solutions.

## Requirements

 - Docker
 - Python 3.6  (for local execution instead of on Docker)

This repository contains the example implementation for how the optimization challenge code will be executed at the end of the competition. Competitors are required to only implement a single method `propose_state` within the file `battery_controller.py`.

This code exists to make it easy for competitors to test their solutions.

## Running the simulation

 1. Clone this repository
 2. Add the data from the competition to the `data` folder. The script expects a folder named `submit` and a file called `metadata.csv` in the `data` directory.
 3. Copy your implementation of `battery_controller.py` into `simulate/battery_controller.py`
 4. Execute the run script: `./run.sh`
 5. Your results will be stored and timestamped in the `all_results` folder each time you execute `run.sh` (Note: `results.csv` in `output` will be overwritten on each subsequent run.)

 **Note for Windows Users: We will accept a pull request with a `run.bat` script that works on Windows machines.**

The only supported execution is within Docker. However, if you want to run the Python code locally rather than in a Docker container, you can still use the `entrypoint.sh` script on unix based system. You may need to install `coreutils` on your system in order to use the `timeout` command. For OSX you can run `brew install coreutils` and then update `entrypoint.sh` to call `gtimeout` instead of `timeout`. Windows users will need to create their own script, although for basic purposes just running `python simulate/simulate.py` should be sufficient.

## Making a submission

#### For the public leaderboard

The public leaderboard just presents the mean of your results so far. Simply submit the most recent results from the `all_results` folder.

#### Code submission

You are required to submit your code in order to be considered for a prize. To do so, you must create a `zip` archive containing ONLY the `assets` folder and the `battery_controller.py` file. These are the only components you may submit. Other folders and files will be ignored. Instructions for submitting are on the competition page.

## Structure of this directory

File | Description
---- | -----
`├── data` | A directory that has all of the input data as `.csv`s that are provided by the competition. **Competitors must add the data themselves after downloading it from the competition.**
`├── output` | A directory for storing the output of a single simulation run.
`├── all_results` | This directory contains results from all of the runs executed.
`├── simulate` | The Python code for the simulation.
`·   ├── assets` | **A FOLDER FOR ANY TRAINED MODELS/DATA THAT NEEDS TO BE LOADED BY `battery_controller.py`**
`·   ├── battery.py` | Contains an object for storing information about the battery.
`·   ├── simulate.py` | Main entrypoint. Controls and executes the simulations.
`·   └── battery_controller.py` | **THIS FILE SHOULD BE IMPLEMENTED BY COMPETITORS**
`├── Dockerfile` | The definition for the Docker container on which the simulation executes.
`├── README.md` | About the project.
`├── entrypoint.sh` | Called inside the container to execute the simulation. Can also be used locally.
`├── requirements.txt` | The Python libraries that will be installed. Only the libraries in this official repo will be available.
`└── run.sh` | The only command you need. Builds and runs simulations in the Docker container.

<a href="https://www.drivendata.org">
    <img src="https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png"/>
</a>
