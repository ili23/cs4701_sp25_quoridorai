#!/bin/bash

# Navigate to the build directory
cd build

# Run make
make

# Set const2 to a fixed value (you can change this as needed)
const2=1000

# Loop through const1 values from 100 to 1000
for const1 in 100 1000 10000 100000
do
    echo "Running with const1=$const1, const2=$const2"
    ./quoridor $const1 $const2 basic_vs_naive.csv
done
