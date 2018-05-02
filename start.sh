#!/bin/bash

# Start Gunicorn processes
echo Starting APP.
exec gunicorn -k gevent -w 1 app:app