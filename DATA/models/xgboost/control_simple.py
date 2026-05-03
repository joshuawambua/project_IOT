#!/usr/bin/env python3
import subprocess
import os
import signal
import sys
import time

PID_FILE = 'predictor.pid'

def start():
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            try:
                os.kill(int(f.read()), 0)
                print("Already running")
                return
            except:
                os.remove(PID_FILE)

    p = subprocess.Popen([sys.executable, 'predictor.py'])
    with open(PID_FILE, 'w') as f:
        f.write(str(p.pid))
    print(f"Started (PID: {p.pid})")

def stop():
    if not os.path.exists(PID_FILE):
        print("Not running")
        return
    with open(PID_FILE) as f:
        pid = int(f.read())
    try:
        os.kill(pid, signal.SIGTERM)
        os.remove(PID_FILE)
        print("Stopped")
    except:
        print("Not found")

def status():
    if not os.path.exists(PID_FILE):
        print("Not running")
        return
    with open(PID_FILE) as f:
        pid = int(f.read())
    try:
        os.kill(pid, 0)
        print(f"Running (PID: {pid})")
    except:
        print("Dead")
        os.remove(PID_FILE)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == 'start':
        start()
    elif cmd == 'stop':
        stop()
    elif cmd == 'status':
        status()
    else:
        print("Usage: python run.py {start|stop|status}")
