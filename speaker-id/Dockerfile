# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# speaker-id service — Phase 1 (ECAPA recognition core, CPU-only).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_SAVEDIR=/models/ecapa

WORKDIR /code

# libsndfile for soundfile (audio decode)
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first so speechbrain doesn't pull the CUDA wheel (~2GB).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio

COPY requirements.txt .
RUN pip install -r requirements.txt

# Phase 4 (Tier 2) diarization is OPT-IN and heavy (pyannote.audio + lightning).
# Keep the default image torch-light; build with --build-arg INSTALL_PYANNOTE=1
# (compose passes it from the INSTALL_PYANNOTE env) to get the real diarizer.
# pyannote.audio>=3.3 loads audio via torchcodec, which needs FFmpeg libs AND a
# torchcodec built for the exact installed torch. The slim base has no FFmpeg, and
# the PyPI torchcodec lags new torch → "Could not load libtorchcodec". Fix both:
# install ffmpeg, and pull torchcodec from the SAME PyTorch CPU index as torch so
# the versions match. (We feed pyannote in-memory waveforms, so this is belt-and-
# suspenders, but it silences the warning and avoids any internal decode path.)
ARG INSTALL_PYANNOTE=0
COPY requirements-diarize.txt .
RUN if [ "$INSTALL_PYANNOTE" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends ffmpeg \
        && rm -rf /var/lib/apt/lists/* \
        && pip install --index-url https://download.pytorch.org/whl/cpu torchcodec \
        && pip install -r requirements-diarize.txt; \
    fi

# Bake the ECAPA weights into the image so the container starts offline and the
# first identify isn't a cold download.
RUN python -c "from speechbrain.inference.speaker import EncoderClassifier; \
EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='/models/ecapa')"

COPY app ./app

EXPOSE 9100

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9100/healthz').status==200 else 1)"

# --proxy-headers + trust forwarded IPs so request.url_for() honors the reverse
# proxy's X-Forwarded-Proto (https) — the OIDC redirect_uri must match the IdP's.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9100", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
