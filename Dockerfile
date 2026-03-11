FROM python:3.11-slim

WORKDIR /app

# Only server.py's dependency — the heavy scientific stack (rasterio, numpy…)
# is for offline Python scripts, not needed at serve time.
RUN pip install --no-cache-dir requests

COPY server.py .
COPY web/ web/

RUN mkdir -p data

EXPOSE 108

CMD ["python", "server.py", "108"]
