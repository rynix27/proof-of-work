# Proof of Work -- container image
#
# ONE process: dashboard + API + background simulation thread (waitress, 8 threads).
# State (event log, replays, leaderboard) lives in /app/data, so do NOT run more
# than one replica/worker against the same volume. See docs/DEPLOYMENT.md.
#
# Secure defaults: non-root user, fails-closed controls (PUBLIC_MODE=1: /control/*
# returns 403 unless CONTROL_TOKEN is set), mock model unless explicitly switched
# to live. Supply secrets at runtime (env / secret manager), never at build time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    PUBLIC_MODE=1 \
    USE_FAKE_MODEL=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --system --create-home --uid 10001 app \
    && mkdir -p /app/data/replays /app/data/training/datasets \
    && chown -R app:app /app

COPY --chown=app:app core ./core
COPY --chown=app:app web ./web
COPY --chown=app:app training ./training
COPY --chown=app:app server.py ./

USER app
VOLUME ["/app/data"]
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','5000'), timeout=3)" || exit 1

STOPSIGNAL SIGTERM
CMD ["python", "server.py"]
