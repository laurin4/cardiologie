FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Runtime writable dirs
RUN mkdir -p /app/outputs /app/data/raw /app/data/processed

# Default interactive shell; run the pipeline explicitly, e.g.:
#   docker run --rm -it <image> \
#     python -m src.pipeline.pipeline --task demo_extraction --reports examples/demo_reports
CMD ["bash"]
