import pytest
import sys
import os

# Add src to sys.path to ensure flaretorch is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
