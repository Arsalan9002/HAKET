# FROM directive instructing base image to build upon
FROM python:3-onbuild
# COPY startup script into known file location in container
COPY start.sh /start.sh

# EXPOSE port 5000 to allow communication to/from server
EXPOSE 8000

RUN ["python", "-m", "spacy", "download", "en"]
# CMD specifcies the command to execute to start the server running.
CMD ["gunicorn", "app:app"]
# done!
