# FROM directive instructing base image to build upon
FROM python:3-onbuild
# COPY startup script into known file location in container
COPY start.sh /start.sh

# EXPOSE port 5000 to allow communication to/from server
EXPOSE 8000

RUN ["chmod", "+x", "/start.sh"]
# CMD specifcies the command to execute to start the server running.
CMD ["/start.sh"]
# done!
