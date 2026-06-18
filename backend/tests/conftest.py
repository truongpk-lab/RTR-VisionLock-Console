"""Pytest setup: always run the suite against the portable default config.

A developer machine may have a config/local.yaml that switches on a GPU deep
stack (SAM2/YOLO/deep ReID). The tests assert the lightweight fallback behaviour,
so we disable the local override for the whole test process.
"""

import os

os.environ.setdefault("RTR_NO_LOCAL_CONFIG", "1")
