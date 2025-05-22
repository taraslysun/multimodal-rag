# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app


# Copy requirements and install
COPY requirements.txt ./
RUN pip install -r requirements.txt
# CMD ["apt-get", "install", "tesseract-ocr", "-y"]

# Install Tesseract OCR
RUN apt-get update && \
    apt-get install -y tesseract-ocr


# Copy the rest of the code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Entrypoint
# CMD ["python", "app/main.py"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
