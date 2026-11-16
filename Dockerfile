# Use a slim Python image to match your local setup
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for some AI/PDF libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command (overridden by GitLab or Compose)
CMD ["python", "scripts/run_data_pipeline.py"]