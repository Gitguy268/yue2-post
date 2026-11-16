#!/bin/sh
# Resumable generation of the 162 train clips, then the 30 eval base clips.
# Finished clips are skipped, so it is safe to stop (Ctrl-C) and start again. Retries up to 20 times if killed.
cd "$(dirname "$0")/.." || exit 1
# caffeinate keeps a Mac awake; it does not exist on Linux (rented GPU box)
awake() { if [ -x /usr/bin/caffeinate ]; then /usr/bin/caffeinate -i "$@"; else "$@"; fi; }
done_count() { ls data/clips/*/item.json 2>/dev/null | wc -l | tr -d ' '; }
for attempt in $(seq 1 20); do
  echo "=== attempt $attempt $(date), $(done_count)/192 clips done ===" | tee -a runs/gen.log
  awake .venv/bin/python -u scripts/gen_data.py train 2>&1 | tee -a runs/gen.log
  awake .venv/bin/python -u scripts/gen_data.py eval-base 2>&1 | tee -a runs/gen.log
  if [ "$(done_count)" -ge 192 ]; then
    echo "=== all generation done $(date) ===" | tee -a runs/gen.log
    exit 0
  fi
  sleep 30
done
exit 1
