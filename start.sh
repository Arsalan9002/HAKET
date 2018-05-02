#!/bin/bash

# Start Gunicorn processes
echo Starting APP.
exec gunicorn --worker-class eventlet -w 1 module:app