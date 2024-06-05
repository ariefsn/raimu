FROM python:3.12-slim

# Update pip
RUN pip install --upgrade pip

# Install HDF5 using apt
RUN apt-get update && apt-get install -y pkg-config gcc libhdf5-dev

# For deepface
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /src

# Install h5py with no-binary flag
RUN pip install --no-binary h5py h5py

COPY ./requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt

COPY ./app /src/app

CMD ["fastapi", "run", "app/main.py", "--port", "80"]