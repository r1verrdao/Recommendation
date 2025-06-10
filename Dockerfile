# Choose base image
FROM python:3.10-slim

# Update and install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# working directory in the container
WORKDIR /app

# Copy files
COPY . /app

# Install python packages
RUN pip install --upgrade pip uv
RUN uv pip install --system -r requirements.txt

# Expose port
EXPOSE 4000

# Run the api.py file
CMD ["python", "api.py"]