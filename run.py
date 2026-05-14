"""Streamlit 데모 실행 launcher.

사용:
    python run.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    app = here / "src" / "landmark_demo" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", "8501"]
    subprocess.run(cmd, cwd=str(here), check=False)
