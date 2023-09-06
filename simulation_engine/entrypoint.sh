#!/bin/bash

timeout 30m python simulate/simulate.py

if [ -f "output/results.csv" ]
then
    echo "Script completed its run."
    echo ""
    cat output/results.csv
    echo ""
    mv output/results.csv /all_results/results_$(date +"%Y-%m-%d-%H_%M_%S").csv
else
    echo "ERROR: Script did execute correctly or timed out."
fi
