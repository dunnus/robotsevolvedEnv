FROM python:3.13-slim
COPY puzzle.json /data/puzzle.json
COPY game.py /opt/robotsevolved/game.py
RUN printf '#!/bin/sh\nexec python /opt/robotsevolved/game.py "$@"\n' > /usr/local/bin/robot-game \
    && chmod +x /usr/local/bin/robot-game && mkdir -p /workspace
WORKDIR /workspace