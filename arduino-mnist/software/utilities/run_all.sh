#!/bin/bash

# Iterate over all files in the data directory
./arduino-interface /dev/ttyUSB0 -p -b -i ./data -o results.csv

# Print out results
python3 utilities/summary.py
