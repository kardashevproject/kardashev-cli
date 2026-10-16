from __future__ import annotations

from .runner import print_report, run_corpus


def main() -> None:
    results = run_corpus()
    print_report(results)


if __name__ == "__main__":
    main()
