#!/usr/bin/env python3
"""
Control Script for XGBoost Predictor v1.0
Usage: python control_v1.py {start|stop|status|restart|logs}
"""

import subprocess
import os
import signal
import sys
import time

PID_FILE = 'predictor_v1.pid'
LOG_FILE = 'predictor_v1.log'

def start():
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            try:
                os.kill(int(f.read()), 0)
                print("❌ Predictor already running")
                return
            except:
                os.remove(PID_FILE)

    print("🚀 Starting XGBoost Predictor v1.0...")
    process = subprocess.Popen(
        [sys.executable, 'predictor_service_v1.py'],
        stdout=open(LOG_FILE, 'w'),
        stderr=subprocess.STDOUT
    )

    with open(PID_FILE, 'w') as f:
        f.write(str(process.pid))

    time.sleep(2)

    # Check if it's actually running
    try:
        os.kill(process.pid, 0)
        print(f"✅ Started (PID: {process.pid})")
        print(f"📝 Log: {LOG_FILE}")
        print("\n📋 Recent log output:")
        os.system(f'type {LOG_FILE} 2>nul | findstr /v "Waiting" | tail -n 10')
    except:
        print("❌ Failed to start - check log file")
        print("\n📋 Log contents:")
        os.system(f'type {LOG_FILE}')

def stop():
    if not os.path.exists(PID_FILE):
        print("❌ Predictor not running")
        return

    with open(PID_FILE, 'r') as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)

        # Force kill if still running
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except:
            pass

        os.remove(PID_FILE)
        print(f"✅ Stopped (PID: {pid})")
    except:
        print("❌ Process not found")
        os.remove(PID_FILE)

def status():
    if not os.path.exists(PID_FILE):
        print("❌ Predictor not running")
        return

    with open(PID_FILE, 'r') as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, 0)
        print(f"✅ RUNNING (PID: {pid})")

        # Show last predictions from log
        if os.path.exists(LOG_FILE):
            print("\n📊 Last predictions:")
            os.system(f'findstr "Prediction" {LOG_FILE} | tail -n 5')
    except:
        print("❌ Stopped")
        os.remove(PID_FILE)

def logs():
    if os.path.exists(LOG_FILE):
        print("\n" + "="*60)
        print("PREDICTOR LOGS")
        print("="*60)
        os.system(f'type {LOG_FILE}')
    else:
        print("No log file found")

def restart():
    print("🔄 Restarting...")
    stop()
    time.sleep(2)
    start()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("="*50)
        print("XGBoost Predictor Control v1.0")
        print("="*50)
        print("Usage: python control_v1.py {start|stop|status|restart|logs}")
        print("")
        print("  start   - Start the predictor service")
        print("  stop    - Stop the predictor service")
        print("  status  - Check service status")
        print("  restart - Restart the service")
        print("  logs    - Show full log file")
        print("="*50)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'start':
        start()
    elif cmd == 'stop':
        stop()
    elif cmd == 'status':
        status()
    elif cmd == 'restart':
        restart()
    elif cmd == 'logs':
        logs()
    else:
        print(f"❌ Unknown command: {cmd}")
