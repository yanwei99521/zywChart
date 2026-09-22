#!/usr/bin/env python3
"""Generate a TradingView Pine script carrying the currently fitted parameters.

TradingView cannot import the dataset JSON directly, but the power-law corridor
only depends on three numbers (genesis, slope, intercept). This renders
``tradingview/bitcoin_power_law.pine`` with the live values substituted, so the
resulting script can be pasted into the Pine editor as-is.

Usage:
    python tools/export_pine.py                       # write generated script
    python tools/export_pine.py --print               # dump to stdout
    python tools/export_pine.py --output out.pine     # custom destination
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chart_service.dataset import build_dataset  # noqa: E402

TEMPLATE = PROJECT_ROOT / "tradingview" / "bitcoin_power_law.pine"
DEFAULT_OUTPUT = PROJECT_ROOT / "tradingview" / "bitcoin_power_law.generated.pine"


def render(root: Path, template: Path) -> str:
    dataset = build_dataset(root)
    model = dataset["model"]
    text = template.read_text(encoding="utf-8")
    return (
        text.replace("{{SLOPE}}", repr(round(float(model["slope"]), 6)))
        .replace("{{INTERCEPT}}", repr(round(float(model["intercept"]), 6)))
        .replace("{{AS_OF}}", str(dataset.get("as_of", "")))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT),
        help="Directory holding data/weekly_model_data.csv (default: project root)",
    )
    parser.add_argument(
        "--template", default=str(TEMPLATE), help="Pine template with {{SLOPE}} placeholders"
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Where to write the generated script"
    )
    parser.add_argument("--print", action="store_true", help="Print instead of writing")
    args = parser.parse_args()

    try:
        script = render(Path(args.root).resolve(), Path(args.template).resolve())
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.print:
        print(script)
        return 0

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
