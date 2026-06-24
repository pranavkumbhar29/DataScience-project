FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose Flask port
EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
