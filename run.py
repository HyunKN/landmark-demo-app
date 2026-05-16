r"""Streamlit 데모 실행 launcher.

사용:
    python run.py
    python run.py --onnx
    python run.py --int8
    python run.py --config .\\config.onnx.toml
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", action="store_true", help="Run with config.onnx.toml")
    parser.add_argument("--int8", action="store_true", help="Run with config.int8.toml")
    parser.add_argument("--config", help="Path to config TOML")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    if args.int8:
        os.environ["LANDMARK_DEMO_CONFIG"] = str(here / "config.int8.toml")
    elif args.onnx:
        os.environ["LANDMARK_DEMO_CONFIG"] = str(here / "config.onnx.toml")
    elif args.config:
        os.environ["LANDMARK_DEMO_CONFIG"] = str(Path(args.config).resolve())

    app = here / "src" / "landmark_demo" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8501",
    ]
    subprocess.run(cmd, cwd=str(here), check=False)
