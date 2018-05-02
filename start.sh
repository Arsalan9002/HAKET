#!/bin/bash

# Start Gunicorn processes
echo Starting APP.
exec python -m spacy download en
exec gunicorn --bind 0.0.0.0:8000 --workers 3 app:app