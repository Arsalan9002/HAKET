#!/bin/bash

# Start Gunicorn processes
echo Spacy download
exec python -m spacy download en
echo Starting Gunicorn.
exec gunicorn --bind 0.0.0.0:5000 --workers 3 app:app