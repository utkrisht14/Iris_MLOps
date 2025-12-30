FROM ubuntu:latest
LABEL authors="utkri"

ENTRYPOINT ["top", "-b"]