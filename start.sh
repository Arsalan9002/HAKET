#!/bin/bash

# Start Gunicorn processes
echo Spacy download
exec python -m spacy download en
echo Starting Gunicorn.
exec gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 module:app