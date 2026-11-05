#!/bin/sh
cd "$(dirname "$0")/.." && uvicorn src.api:app --host 127.0.0.1 --port 8000
