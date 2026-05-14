# Base image — Python 3.10 slim version
# 'slim' means smaller image size — removes unnecessary system packages
FROM python:3.10-slim

# Set working directory inside the container
# All subsequent commands run from /app
WORKDIR /app

# Copy requirements file first
# Docker caches this layer separately — if requirements don't change,
# Docker skips reinstalling packages on next build (faster rebuilds)
COPY requirements.txt .

# Install all Python dependencies
# --no-cache-dir reduces image size by not storing pip cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy Gradio app file
COPY app.py .

# Copy FastAPI backend folder
COPY api/ ./api/

# Tell Docker this container listens on port 7860
# HuggingFace Spaces requires port 7860 specifically
EXPOSE 7860

# Environment variable — model loads from HuggingFace Hub
# Not from local folder — keeps container size small
ENV MODEL_PATH="aniqua-nawar/banglamind-emotion"

# Start command — runs when container launches
# Starts the Gradio app which listens on port 7860
CMD ["python", "app.py"]