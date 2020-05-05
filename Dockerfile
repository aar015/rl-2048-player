# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM alpine:latest

# Here we get all system packages.
RUN apk add --no-cache python3 nginx ca-certificates py3-numpy py3-flask py3-gevent py3-gunicorn

# Here we get all python packages.

# Set some environment variables.
# PYTHONUNBUFFERED keeps Python from buffering our standard output stream, which means that logs can be delivered to the user quickly.
# PYTHONDONTWRITEBYTECODE keeps Python from writing the .pyc files which are unnecessary in this case.
# We also update PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY program /opt/program
WORKDIR /opt/program
