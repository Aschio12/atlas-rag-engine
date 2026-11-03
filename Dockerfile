FROM python:3.12-slim
WORKDIR /srv
RUN useradd --create-home --uid 1000 atlas \
 && mkdir -p /data/blobs && chown -R atlas /data
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .
# Optional OCR for scanned PDFs: docker build --build-arg WITH_OCR=true .
ARG WITH_OCR=false
RUN if [ "$WITH_OCR" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends tesseract-ocr \
      && rm -rf /var/lib/apt/lists/* \
      && pip install --no-cache-dir ".[ocr]"; \
    fi
ENV STORAGE_DIR=/data/blobs
VOLUME /data
USER atlas
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
