#!/usr/bin/env python
"""Quick endpoint verification."""
import subprocess
import time
import sys
import requests

# Start server
print("[Server] Starting FastAPI server on port 8000...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.server:app", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for server
time.sleep(8)

try:
    # Test /agent endpoint
    print("\n[Test] Checking /agent endpoint...")
    resp = requests.get("http://localhost:8000/agent", timeout=5)
    if resp.status_code == 200:
        if "<!DOCTYPE" in resp.text or "<html" in resp.text:
            print(f"[Test] PASS /agent returns HTML ({len(resp.text)} bytes)")
        else:
            print(f"[Test] FAIL /agent returned non-HTML (first 100 chars: {resp.text[:100]})")
    else:
        print(f"[Test] FAIL /agent returned {resp.status_code}: {resp.text[:200]}")

    # Test /rag endpoint
    print("[Test] Checking /rag endpoint...")
    resp = requests.get("http://localhost:8000/rag", timeout=5)
    if resp.status_code == 200:
        if "<!DOCTYPE" in resp.text or "<html" in resp.text:
            print(f"[Test] PASS /rag returns HTML ({len(resp.text)} bytes)")
        else:
            print(f"[Test] FAIL /rag returned non-HTML")
    else:
        print(f"[Test] FAIL /rag returned {resp.status_code}")

    # Test /api/health (should require auth)
    print("[Test] Checking /api/health (should require auth)...")
    resp = requests.get("http://localhost:8000/api/health", timeout=5)
    if resp.status_code in [401, 403]:
        print(f"[Test] PASS /api/health correctly requires auth ({resp.status_code})")
    else:
        print(f"[Test] FAIL /api/health returned {resp.status_code} (expected 401/403)")

    print("\n[Test] PASS Endpoint verification complete!")

except Exception as e:
    print(f"[Test] ERROR: {e}")
    sys.exit(1)

finally:
    # Kill server
    proc.terminate()
    proc.wait(timeout=5)
    print("[Server] Cleaned up")
