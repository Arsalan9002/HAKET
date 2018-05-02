# FROM directive instructing base image to build upon
FROM python:3-onbuild

# EXPOSE port 5000 to allow communication to/from server
EXPOSE 8000
EXPOSE 5000

RUN ["python", "-m", "spacy", "download", "en"]
# CMD specifcies the command to execute to start the server running.
CMD ["gunicorn", "-k", "gevent", "-w", "1", "app:app"]

# done!
