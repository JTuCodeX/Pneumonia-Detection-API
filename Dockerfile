# Use lightweight Python image
FROM python:3.9.6

# Set working directory
WORKDIR /app

# Install system dependencies (needed for Pillow, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt


# Copy the whole app
COPY . .

# Expose port
EXPOSE 8000

# Run with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
