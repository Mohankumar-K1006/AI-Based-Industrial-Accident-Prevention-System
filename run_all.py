"""
Root launcher redirecting to industrial-accident-prevention/run_all.py
"""
import os
import sys

pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "industrial-accident-prevention")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)
os.chdir(pkg_dir)

import run_all

if __name__ == "__main__":
    run_all.main()
