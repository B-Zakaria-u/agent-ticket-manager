FROM python:3.13.9-slim

# Install system dependencies (git for gitpython and PyGithub)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the API port that the application runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
