#!/bin/bash

# build the container
docker build -t power-laws-optimization/simulate .

# limit container to 1 cpu, 4g RAM, no swap
docker run --cpus="1" \
           --memory="4g" \
           --memory-swap="4g" \
           --mount type=bind,source="$(pwd)"/all_results,target=/all_results \
           power-laws-optimization/simulate
