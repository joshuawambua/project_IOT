"""
Control Script - Start, Stop, Restart, Status
Usage:
    python control.py start
    python control.py stop
    python control.py restart
    python control.py status
"""
import subprocess
import psutil
import os
import sys
import json
from datetime import datetime

PID_FILE = 'predictor.pid'

def find_process():
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
            if 'predictor_service.py' in cmd and 'python' in cmd:
                return proc
        except:
            continue
    return None

def start():
    proc = find_process()
    if proc:
        print(f"❌ Already running (PID: {proc.pid})")
        return

    print("🚀 Starting predictor...")
    process = subprocess.Popen(
        [sys.executable, 'predictor_service.py'],
        creationflags=subprocess.CREATE_NO_WINDOW
    )
    with open(PID_FILE, 'w') as f:
        f.write(str(process.pid))
    print(f"✓ Started (PID: {process.pid})")
    print("✓ Log: predictor.log")

def stop():
    proc = find_process()
    if not proc:
        print("❌ Not running")
        return

    print(f"🛑 Stopping (PID: {proc.pid})...")
    proc.terminate()
    proc.wait(timeout=5)
    print("✓ Stopped")
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

def restart():
    print("🔄 Restarting...")
    stop()
    import time
    time.sleep(2)
    start()

def status():
    proc = find_process()
    print("=" * 40)
    print("XGBOOST PREDICTOR STATUS")
    print("=" * 40)
    if proc:
        print(f"✅ RUNNING (PID: {proc.pid})")
        if os.path.exists(PID_FILE):
            with open(PID_FILE, 'r') as f:
                saved_pid = f.read().strip()
                print(f"📌 Saved PID: {saved_pid}")
    else:
        print("❌ STOPPED")
    print("=" * 40)

def main():
    if len(sys.argv) < 2:
        print("Usage: python control.py {start|stop|restart|status}")
        return

    cmd = sys.argv[1].lower()
    if cmd == 'start': start()
    elif cmd == 'stop': stop()
    elif cmd == 'restart': restart()
    elif cmd == 'status': status()
    else: print(f"Unknown: {cmd}")

if __name__ == "__main__":
    main()
