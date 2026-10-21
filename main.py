"""
Root entry point redirecting to industrial-accident-prevention/main.py
Allows running 'python main.py' directly from the repository root.
"""
import os
import sys

package_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "industrial-accident-prevention")
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)
os.chdir(package_dir)

import main

if __name__ == "__main__":
    main.main()
