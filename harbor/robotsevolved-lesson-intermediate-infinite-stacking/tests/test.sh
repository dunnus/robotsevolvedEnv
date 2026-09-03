#!/bin/sh
set -eu
mkdir -p /logs/verifier
reward="$(python /opt/robotsevolved/game.py evaluate /workspace/solution.json 2>/tmp/verifier-error || true)"
case "$reward" in 0|0.0|0.00|0.000|"") reward="0.0" ;; esac
printf '%s\n' "$reward" > /logs/verifier/reward.txt
cat /tmp/verifier-error >&2 2>/dev/null || true