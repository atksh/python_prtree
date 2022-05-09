FROM python:3.10-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG USERNAME=user
ARG GROUPNAME=user
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME
RUN apt-get update && apt-get install -y git openssh-client vim wget curl

USER $USERNAME
ENV PATH $PATH:/home/user/.local/bin
WORKDIR /code
