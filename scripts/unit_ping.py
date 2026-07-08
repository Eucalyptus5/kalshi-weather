import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ws_watchdog import post_ntfy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="notify when a long-running unit stops")
    parser.add_argument("--unit", required=True, help="unit that stopped; systemd passes %%n")
    return parser


def run(args: argparse.Namespace) -> int:
    result = os.environ.get("SERVICE_RESULT", "none")
    exit_code = os.environ.get("EXIT_CODE", "none")
    exit_status = os.environ.get("EXIT_STATUS", "none")
    verdict = "success" if result == "success" else "failed"
    message = (
        f"unit_stopped verdict={verdict} unit={args.unit} "
        f"result={result} exit_code={exit_code} exit_status={exit_status}"
    )

    topic = os.environ.get("KW_WATCHDOG_NTFY_TOPIC", "")
    if topic:
        try:
            post_ntfy(topic, [message])
            ntfy = "sent"
        except OSError:
            ntfy = "failed"
    else:
        ntfy = "skipped"

    print(f"{message} ntfy={ntfy}")
    return 1 if ntfy == "failed" else 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
