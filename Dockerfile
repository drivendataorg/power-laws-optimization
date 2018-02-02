FROM continuumio/anaconda3:latest

# requirements first so they are cached
RUN mkdir /simulation
COPY ./requirements.txt /simulation/requirements.txt

WORKDIR /simulation
RUN pip install -r requirements.txt

# For storing results from multiple runs
VOLUME ["/all_results"]

# get latest code
ADD . /simulation
RUN chmod +x entrypoint.sh

# CMD ["timeout", "10s", "sleep", "20s"]
# CMD ["timeout", "10s", "python", "simulate/simulate.py"]
ENTRYPOINT /simulation/entrypoint.sh
