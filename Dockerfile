# main stage
FROM --platform=$TARGETPLATFORM python:3.6.9-slim AS main

ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends vim zip unzip libgl1-mesa-glx libgl1 libglib2.0-0 libgomp1 libegl1

# add python package
RUN pip3 install --upgrade pip
RUN --mount=type=bind,source=/wheel,target=/mnt/pypi \
    --mount=type=bind,source=/requirements.txt,target=/mnt/requirements.txt \
    pip3 install --no-cache-dir -r /mnt/requirements.txt -f /mnt/pypi/
RUN echo "/usr/lib/python3.6/dist-packages" > /usr/local/lib/python3.6/site-packages/tensorrt.pth

# app
ARG TARGETARCH
ADD app.tar /app

WORKDIR /app
ENV PYTHONPATH="/app/"
ENV LC_ALL="C.UTF-8" LANG="C.UTF-8"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

CMD ["sh", "-c", "gunicorn svc:app --workers ${GUNICORN_W:-1} --threads ${GUNICORN_T:-1} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-5124} --timeout 300"]
# CMD ["sh", "-c", "gunicorn svc:app -w ${GUNICORN_W:-1} --threads 1 -b 0.0.0.0:${PORT:-5119} --timeout 300 --log-level 'debug'"]
