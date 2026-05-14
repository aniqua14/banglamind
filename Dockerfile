# Start from official Python image
# We use 3.10-slim — smaller than full Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first
# Docker caches this layer — only reinstalls if requirements.txt changes
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of application code
COPY api/ ./api/

# Create models directory
RUN mkdir -p models/banglabert-emotion-model

# Expose port 7860
# HuggingFace Spaces expects port 7860 specifically
EXPOSE 7860

# Environment variable — tells the app where to find the model
ENV MODEL_PATH="aniqua-nawar/banglamind-emotion"
ENV PORT=7860

# Start the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]