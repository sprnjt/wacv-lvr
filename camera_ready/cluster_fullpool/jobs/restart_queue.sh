#!/bin/bash
# One-off: restart the queue process so it picks up code changes (idempotent via a marker file). Used by job restart_queue1.
cd $HOME/CultureHaystack_handoff/fullpool
[ -f logs/restart_queue1.done ] && exit 0
touch logs/restart_queue1.done
pkill -f ext/jobqueue.py
sleep 3
setsid nohup python3 ext/jobqueue.py > logs/queue.out 2>&1 < /dev/null &
sleep 5
echo "- $(date +%F\ %T) [fix] queue restarted: backfill enabled (a later job may start while a larger job waits for GPU memory; another tenant holds ~17 GB)" >> RUN_LOG.md
