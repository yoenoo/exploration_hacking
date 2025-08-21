# ps auxf

# 1) Find a busy worker and get its process group id (PGID)
PGID=$(ps -o pgid= -p $1 | tr -d ' ')

# 2) Try graceful shutdown of the entire process group
kill -TERM -$PGID
sleep 5

# 3) If still alive, force kill the whole group
kill -KILL -$PGID