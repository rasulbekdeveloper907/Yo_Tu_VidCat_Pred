# ============================
# Base image (ML safe)
# ============================
FROM python:3.11-slim

# ============================
# Environment variables
# ============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================
# System dependencies
# ============================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================
# Working directory
# ============================
WORKDIR /app

# ============================
# Python dependencies
# ============================
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ============================
# Copy project files
# ============================
COPY demo/ demo/
COPY Models/ Models/

# ============================
# Security: non-root user
# ============================
RUN useradd -m appuser
USER appuser

# ============================
# Gradio port
# ============================
EXPOSE 7860

# ============================
# Run Gradio app
# ============================
CMD ["python", "demo/app.py"]
