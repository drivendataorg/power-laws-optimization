FROM continuumio/anaconda3:latest

RUN apt-get update && apt-get install -y build-essential

# Install requirements first so they are cached
RUN mkdir /simulation
COPY ./requirements.txt /simulation/requirements.txt

WORKDIR /simulation
RUN pip install -r requirements.txt

# Use this volume for storing results from multiple runs during development
VOLUME ["/all_results"]

# Copy the latest code to the container
ADD . /simulation

RUN chmod +x entrypoint.sh

# Execute the entrypoint.sh script inside the container when we do docker run
ENTRYPOINT ["/simulation/entrypoint.sh"]
