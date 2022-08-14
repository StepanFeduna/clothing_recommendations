FROM python:3.8.13-slim-buster

# Maintainer info
LABEL maintainer="feduna.stepan@gmail.com"

# set working directory
WORKDIR /usr/src/clothing_recommendations

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
  && apt-get -y install netcat gcc postgresql \
  && apt-get clean

# install python dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# add app
COPY . .

CMD ["python", "main.py"]