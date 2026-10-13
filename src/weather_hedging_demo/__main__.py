import argparse
import json
from pathlib import Path

from .run import run, verify


def main():
    parser = argparse.ArgumentParser(description="Synthetic degree-day tracking demo")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("output", type=Path)
    runner.add_argument("--seed", type=int, default=7)
    checker = commands.add_parser("verify")
    checker.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.output, seed=args.seed) if args.command == "run" else verify(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
