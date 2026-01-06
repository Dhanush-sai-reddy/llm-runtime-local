# Use a slim python image to keep size down
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir huggingface_hub

# Copy the script
COPY downloader.py .

# Create the data directory
RUN mkdir /data

# Default command (can be overridden)
ENTRYPOINT ["python", "downloader.py"]