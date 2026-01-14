# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create a non-root user for security (and HF Spaces compatibility)
RUN useradd -m -u 1000 user

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Create persistent storage directory and set permissions
# In HF Spaces, /data is often a persistent mount point if configured, 
# or we just use a local folder.
RUN mkdir -p /app/data && chown -R user:user /app/data

# Switch to non-root user
USER user

# Expose Hugging Face default port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
