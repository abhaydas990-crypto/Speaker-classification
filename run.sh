#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt; }
source .venv/bin/activate
python app.py
