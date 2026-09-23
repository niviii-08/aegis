"""End-to-End Test for AEGIS Production Backend.

Usage::
    python scripts/e2e_test.py
"""

import logging
import requests
import sys
import time
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError("Could not locate AEGIS project root")

PROJECT_ROOT = find_project_root()

def wait_for_server(url: str, timeout: int = 15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False

def main():
    logger.info("Starting FastAPI server for E2E testing...")
    
    # Start server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"],
        cwd=str(PROJECT_ROOT)
    )
    
    try:
        base_url = "http://localhost:8000"
        logger.info(f"Waiting for server at {base_url}/health ...")
        if not wait_for_server(f"{base_url}/health"):
            logger.error("Server did not start in time.")
            return 1
            
        logger.info("Server is up. Running E2E tests...")
        
        # Test 1: Health
        r = requests.get(f"{base_url}/health")
        assert r.status_code == 200
        logger.info("✅ Health endpoint passed.")
        
        logger.info("E2E API test completed successfully. (File inference skipped in basic script)")
        
    finally:
        server_process.terminate()
        server_process.wait()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
