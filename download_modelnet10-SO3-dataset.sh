#!/bin/bash

# Define the directory paths
data_dir="./data"
dataset_dir="${data_dir}/ModelNet10-SO3"

# Function to get the size of a directory in GB
get_dir_size() {
    du -sb $1 | awk '{print $1/1024/1024/1024}'
}

# Check if the dataset directory exists and is less than 2GB
if [[ ! -d "${dataset_dir}" || $(get_dir_size "${dataset_dir}") < 2 ]]; then
    # Create the data directory if it doesn't exist
    mkdir -p "${data_dir}"
    
    # Change to the data directory
    cd "${data_dir}"
    
    # Download and unzip the dataset
    wget -q https://i81server.iar.kit.edu/RotationNormFlow/ModelNet10-SO3.zip
    unzip ModelNet10-SO3.zip -d ModelNet10-SO3
    rm ModelNet10-SO3.zip
fi
