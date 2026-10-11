"""Compatibility entry point; use the documented autoqa.py subcommands."""

from autoqa import main


if __name__ == "__main__":
    raise SystemExit(main())
