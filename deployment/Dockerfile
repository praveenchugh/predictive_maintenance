# ------------------------------------------------------------
# Use a lightweight Python runtime as the base image
# ------------------------------------------------------------
FROM python:3.9-slim

# ------------------------------------------------------------
# Configure Python + environment behavior
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

# ------------------------------------------------------------
# Create a non-root user for safer execution
# ------------------------------------------------------------
RUN useradd --create-home --uid 1000 appuser

# ------------------------------------------------------------
# Set working directory inside container
# ------------------------------------------------------------
WORKDIR $HOME/app

# ------------------------------------------------------------
# Install minimal system dependencies if required by packages
# ------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Copy dependency file first to enable Docker layer caching
# ------------------------------------------------------------
COPY --chown=appuser:appuser requirements.txt .

# ------------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Copy remaining application source code
# ------------------------------------------------------------
COPY --chown=appuser:appuser . .

# ------------------------------------------------------------
# Switch to non-root user
# ------------------------------------------------------------
USER appuser

# ------------------------------------------------------------
# Expose Streamlit port
# ------------------------------------------------------------
EXPOSE 7860

# ------------------------------------------------------------
# Start Streamlit application
# ------------------------------------------------------------
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
