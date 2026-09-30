# Use a lightweight base image with Miniforge installed
FROM condaforge/miniforge3:latest

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# Install essential build tools (git is required to clone dependencies)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the environment file first (to leverage Docker layer caching)
COPY environment.yml .

# Create the conda environment
RUN mamba env create -f environment.yml -y && conda clean -afy

# Copy the rest of the OptiCut repository
COPY . .

# Run the installation script inside the Conda environment
# We use bash -c with "source activate" to ensure the environment is loaded
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate opticut-env && \
    chmod +x install_opticut.sh && \
    ./install_opticut.sh"

# Set the default command to activate the environment and run the tests
# This proves to the reviewer that the installation is successful
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate opticut-env && pytest tests/"]
