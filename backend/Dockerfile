# Use official Python image
FROM python:3.11-slim

WORKDIR /app


COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y tesseract-ocr


COPY . .

EXPOSE 8000

# Entrypoint
# CMD ["python", "app/main.py"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
