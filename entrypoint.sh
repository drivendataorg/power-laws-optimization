#!/bin/bash

timeout 3s python simulate/simulate.py

if [ -f "output/results.csv" ]
then
    echo "Script completed its run."
    echo ""
    cat output/results.csv
    echo ""
    mv output/results.csv /all_results/results_$(date +"%Y-%m-%d-%H:%M:%S").csv
else
    echo "ERROR: Script timed out."
fi
