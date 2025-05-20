# Stage 1: Install dependencies (pip packages)
FROM tensorflow/tensorflow:2.15.0-gpu AS builder

RUN pip install --upgrade pip

# Install apt dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    pkg-config \
    gcc \
    libhdf5-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY ./requirements.txt /src/requirements.txt

# Install h5py no-binary to avoid wheel issues
RUN pip install --no-binary=h5py h5py

# Install python deps excluding tensorflow to avoid reinstall
RUN grep -vE "^(tensorflow|tensorflow-gpu|tensorflow-cpu)" /src/requirements.txt > /src/requirements-no-tf.txt \
 && pip install --no-cache-dir --upgrade --ignore-installed -r /src/requirements-no-tf.txt

# Stage 2: Final image, copy installed packages + source
FROM tensorflow/tensorflow:2.15.0-gpu

# Install runtime dependencies (libGL etc) in final image too
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libgl1-mesa-glx \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
# Copy apt dependencies from builder stage (optional, because base image should be same)
# Here we rely on base image for CUDA, so no apt install again

WORKDIR /src

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app source code
COPY ./app /src/app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
