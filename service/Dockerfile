# syntax=docker/dockerfile:1
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
# The GPU runtime and its CUDA libraries are about 1.5 GB. When their wheels are in ./wheels (fetch them
# once with `pip download`; the folder is not in git) they are installed from there and pip downloads only
# the small rest. Without the folder everything comes from PyPI as usual.
RUN --mount=type=bind,source=.,target=/src \
    if ls /src/wheels/*.whl >/dev/null 2>&1; then pip install --no-cache-dir --no-index --no-deps /src/wheels/*.whl; fi \
 && pip install --no-cache-dir -r requirements.txt
COPY tracker_service ./tracker_service

# No footage is stored: the service has nowhere to write, and nothing in it writes.
ENV PYTHONUNBUFFERED=1 ORT_DISABLE_TELEMETRY=1 TRACKER_HOST=0.0.0.0 TRACKER_PORT=8791 TRACKER_MODELS=/models
USER 1000:1000
EXPOSE 8791
CMD ["python", "-m", "tracker_service"]
