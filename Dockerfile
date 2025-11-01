# ===== Stage 1: Base Image =====
FROM python:3.11-slim

# Working directory inside container
WORKDIR /app

# Install OS deps (if needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model files
COPY app_folder/ ./app_folder/
COPY models/ ./models/

# Expose FastAPI default port
EXPOSE 8000

# Optional healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8000/docs || exit 1

# Default command — run FastAPI backend
# (You can override this at runtime to run Streamlit)
CMD ["uvicorn", "app_folder.api:app", "--host", "0.0.0.0", "--port", "8000"]
