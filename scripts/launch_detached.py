"""Start a command in its own session so it survives the Claude session ending. usage: launch_detached.py LOG cmd..."""
import subprocess, sys
log = open(sys.argv[1], "ab")
p = subprocess.Popen(["caffeinate", "-i", *sys.argv[2:]], stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
print(p.pid)
