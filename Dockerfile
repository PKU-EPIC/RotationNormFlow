# Use a version I tested
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install necessary system packages for compilation
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git g++ libgl1-mesa-glx libglib2.0-0 unzip wget

# Install pytorch3d from Git repo
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

WORKDIR /workspace
