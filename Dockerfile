FROM  public.ecr.aws/docker/library/python:3.11.10-slim-bookworm

RUN  <<eot
    set -ex

    pip install --upgrade pip
    pip install psutil ipython pandas numpy matplotlib plotly
eot
