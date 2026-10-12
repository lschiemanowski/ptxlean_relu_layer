#!/usr/bin/env python3
"""Prepare dependencies, build the Sequential bridge, and audit its theorem."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    env.setdefault('LEAN_NUM_THREADS', '4')
    commands = [
        ['lake', 'env', sys.executable, 'scripts/prepare_dependencies.py'],
        ['lake', 'build', 'ReluLayer.SequentialBridge'],
        ['lake', 'env', 'lean', '--run', 'checks/SequentialBridge.lean'],
    ]
    for command in commands:
        print(f"Running: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
