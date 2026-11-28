"""Optional combined evaluation entry point.

Individual suite entry points should be imported and called from their own
primary modules. This module may provide a small ``main()`` to invoke them
together after the suites are implemented.
"""


def main() -> int:
    """Return without running evaluation until suite modules are implemented."""
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
