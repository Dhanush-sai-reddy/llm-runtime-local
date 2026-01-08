
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir huggingface_hub

COPY downloader.py .

RUN mkdir /data

ENTRYPOINT ["python", "downloader.py"]