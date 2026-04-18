FROM python:3.11-slim-bookworm

LABEL maintainer="arm"
LABEL description="Temperature & Humidity forecasting service (Bidirectional LSTM)"

ENV TZ=Asia/Bangkok
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# ซ่อน TensorFlow info/warning logs
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev \
        libfreetype6 \
        libhdf5-dev \
        tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────
COPY requirements.txt .
# --timeout=300 สำหรับ tensorflow wheel ขนาดใหญ่ (~500MB) บน Raspi
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --timeout=300 -r requirements.txt

# ── Application source ───────────────────────────────────────────────────
COPY src/     src/
COPY models/  models/
COPY config/  config/

# ── Healthcheck ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "from src.service.app import run_once; print('ok')" || exit 1

# ── Default command ──────────────────────────────────────────────────────
CMD ["python", "-m", "src.service.app", "--at", "00:00"]
