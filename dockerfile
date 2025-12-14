# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirement file into the container at /app
COPY requirements.txt /app/

# Install any exact package specified in requirement.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Set python path so it can find the src module
ENV PYTHONPATH=/app

# when the container launches
CMD ["python", "src/inference.py"]
