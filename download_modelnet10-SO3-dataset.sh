#!/bin/bash

# Define the directory paths
data_dir="./data"

# Function to get the size of a directory in GB
get_dir_size() {
    du -sb $1 | awk '{print $1/1024/1024/1024}'
}

# Check if the dataset directory exists and is less than 2GB
if [[ ! -d "${data_dir}" || $(get_dir_size "${data_dir}") < 2 ]]; then
    # Create the data directory if it doesn't exist
    mkdir -p "${data_dir}"

    # Check if ModelNet10-SO3.zip is present in the current directory
    if [[ -f "ModelNet10-SO3.zip" ]]; then
        # Move the zip file to the data directory
        mv ModelNet10-SO3.zip "${data_dir}/"
        cd "${data_dir}"
    else
        # Change to the data directory
        cd "${data_dir}"
    
        # Download the dataset
        wget https://i81server.iar.kit.edu/RotationNormFlow/ModelNet10-SO3.zip
    fi
    
    # Unzip the dataset
    unzip ModelNet10-SO3.zip

    # Remove the zip file
    rm ModelNet10-SO3.zip
    cd ..
fi