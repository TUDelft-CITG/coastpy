import os
import sys

# Hacky solution for vscode debugger in modular mode.  
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# read version from installed package
# from importlib.metadata import version

# __version__ = version("coastpy")

