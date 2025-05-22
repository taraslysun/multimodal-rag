# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app


# Copy requirements and install
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Entrypoint
CMD ["python", "main.py"]
