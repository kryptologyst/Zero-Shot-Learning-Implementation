#!/usr/bin/env python3
"""Simple run script for zero-shot learning demo."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main function."""
    print("🔍 Zero-Shot Learning Implementation")
    print("=" * 50)
    print()
    print("Available commands:")
    print("  demo        - Run basic demo")
    print("  train       - Run training pipeline")
    print("  streamlit   - Launch Streamlit demo")
    print("  test        - Run tests")
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python run.py <command>")
        print("Example: python run.py demo")
        return
    
    command = sys.argv[1].lower()
    
    if command == "demo":
        print("Running demo...")
        try:
            from scripts.demo import run_demo
            run_demo()
        except ImportError as e:
            print(f"CLIP not available: {e}")
            print("Running simple demo without CLIP...")
            from scripts.simple_demo import run_simple_demo
            run_simple_demo()
    
    elif command == "train":
        print("Running training pipeline...")
        print("Note: This requires CLIP model to be available")
        try:
            from src.train.trainer import main as train_main
            train_main()
        except Exception as e:
            print(f"Training failed: {e}")
            print("Make sure you have installed all dependencies and CLIP model is available")
    
    elif command == "streamlit":
        print("Launching Streamlit demo...")
        import subprocess
        subprocess.run(["streamlit", "run", "demo/streamlit_app.py"])
    
    elif command == "test":
        print("Running tests...")
        import subprocess
        result = subprocess.run([sys.executable, "tests/test_basic_simple.py"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: demo, train, streamlit, test")

if __name__ == "__main__":
    main()
