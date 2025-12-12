# Base image with python version = python virtual env
# (standard version = python 3.12)

FROM python:3.10.6-buster
WORKDIR /ReefSight


# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project logic
COPY project_logic/ project_logic/
COPY api/ api/

# Copy the models
COPY models/ models/


# Make directories that we need, but that are not included in the COPY
RUN mkdir /raw_data

# Set default port
ENV PORT=8000

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT


# Install HDF5 library
#RUN apt-get update && apt-get install -y \
 #   libhdf5-dev \
  #  && apt-get clean
