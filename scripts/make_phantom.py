"""Write a synthetic phantom case (developer smoke test only):  python scripts/make_phantom.py data/raw/phantom"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.phantom import write_case

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/phantom"
    print(write_case(out, name="phantom", shape=(96, 112, 80), noise=0.06))
