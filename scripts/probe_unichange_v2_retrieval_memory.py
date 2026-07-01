import sys
from pathlib import Path
from ucv2_memory_core import run

run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
