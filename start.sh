#!/bin/bash

# Start Gunicorn processes
echo Starting APP.
exec python -m spacy download en && python app.py