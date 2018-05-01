#!/bin/bash

# Start Gunicorn processes
echo Starting APP.
exec python -m spacy download en
exec python app.py