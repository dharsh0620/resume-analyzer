#!/usr/bin/env bash
python -m nltk.downloader punkt stopwords >/dev/null 2>&1 || true
gunicorn app:app --bind 0.0.0.0:$PORT
