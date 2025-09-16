FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

# Use the Conda installation that already exists in the base image
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget bzip2 vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure Python 3.7 is installed in the existing Conda environment
RUN conda install -y python=3.7 && conda clean -afy

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app/

# Install Python dependencies
# Install Python dependencies with compatible setuptools
RUN pip install --upgrade "setuptools<65" "setuptools_scm<6" && \
    pip install torch==1.4.0 transformers==2.10.0 seqeval==0.0.12


# Expose a port for potential API communication
EXPOSE 8080

# Default command
CMD ["/bin/bash"]
