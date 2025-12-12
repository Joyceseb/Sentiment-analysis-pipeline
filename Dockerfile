# 1. Base Python image
FROM python:3.10-slim

# 2. Define working directory
WORKDIR /app

# 3. Copy project files into the container
COPY . .

# 4. Install dependencies and requirements
RUN pip install --no-cache-dir -r requirements.txt

# 5. Create environment variable for Python path
ENV PYTHONPATH=/app

# 6. Expose the port the app runs on
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 
