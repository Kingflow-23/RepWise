"""Run several evaluation suites in one command.

Each suite is normally run from its own module:

    python -m src.evaluation.retrieval
    python -m src.evaluation.generation
    python -m src.evaluation.red_team

This entry point only chains them, e.g. ``python -m src.evaluation.main
retrieval red_team`` (no argument runs all three). Every suite writes its own
``report.md`` and ``results.json``; there is no combined benchmark.
"""

import argparse

from ..observability.observability import configure_logging
from .generation import run_generation
from .red_team import run_red_team
from .retrieval import run_retrieval

SUITES = {
    "retrieval": run_retrieval,
    "generation": run_generation,
    "red_team": run_red_team,
}


def main(argv: list[str] | None = None) -> int:
    """Run the selected suites in order and print each report folder."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("suites", nargs="*", choices=list(SUITES), metavar="SUITE")
    parser.add_argument("--label", default="", help="suffix for the run folders")
    args = parser.parse_args(argv)

    configure_logging("evaluation")
    for name in args.suites or SUITES:
        print(f"{name}: {SUITES[name](label=args.label)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
