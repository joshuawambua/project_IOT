import subprocess
import psutil
import os
import sys

PID_FILE = 'pm25_predictor.pid'

def find_process():
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
            if 'pm25_predictor_service.py' in cmd and 'python' in cmd:
                return proc
        except:
            continue
    return None

def start():
    proc = find_process()
    if proc:
        print(f"Already running (PID: {proc.pid})")
        return
    print("Starting PM2.5 Predictor...")
    process = subprocess.Popen([sys.executable, 'pm25_predictor_service.py'], creationflags=subprocess.CREATE_NO_WINDOW)
    with open(PID_FILE, 'w') as f:
        f.write(str(process.pid))
    print(f"Started (PID: {process.pid})")

def stop():
    proc = find_process()
    if not proc:
        print("Not running")
        return
    print(f"Stopping...")
    proc.terminate()
    proc.wait(timeout=5)
    print("Stopped")
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

def status():
    proc = find_process()
    if proc:
        print(f"RUNNING (PID: {proc.pid})")
    else:
        print("STOPPED")

def main():
    if len(sys.argv) < 2:
        print("Usage: python control_pm25.py {start|stop|status}")
        return
    cmd = sys.argv[1].lower()
    if cmd == 'start': start()
    elif cmd == 'stop': stop()
    elif cmd == 'status': status()
    else: print(f"Unknown: {cmd}")

if __name__ == "__main__":
    main()
