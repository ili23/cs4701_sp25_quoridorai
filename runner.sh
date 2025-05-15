#!/bin/bash

# Navigate to the build directory
cd second_attempt/build

# Run make
make

# Set const2 to a fixed value (you can change this as needed)
const2=500

# Loop through const1 values from 100 to 1000
for const1 in {100..1000..100}
do
    echo "Running with const1=$const1, const2=$const2"
    ./quoridor $const1 $const2
done
