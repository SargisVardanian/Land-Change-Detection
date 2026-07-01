import sys
from pathlib import Path
from ucv2_ddp_core import run

resume = Path(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] else None
code = run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), resume)
assert code == 0
