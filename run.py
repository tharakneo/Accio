import subprocess
import sys
import os
import signal
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "frontend")

procs = []

def shutdown(sig=None, frame=None):
    print("\nShutting down...")
    for p in procs:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.api.main:app", "--reload", "--port", "8000"],
    cwd=ROOT,
)
procs.append(backend)

frontend = subprocess.Popen(
    ["npm", "run", "dev"],
    cwd=FRONTEND,
)
procs.append(frontend)

print("Backend:  http://localhost:8000")
print("Frontend: http://localhost:5173")
print("Ctrl+C to stop\n")

for p in procs:
    p.wait()
