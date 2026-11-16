"""Read real Yosys stat -json output; no fallback resource estimates."""
import argparse
import json
from pathlib import Path


def parse_report(log_file="yosys_stats.json"):
    report = json.loads(Path(log_file).read_text(encoding="utf-8"))
    if "design" in report:
        cells = report["design"]["num_cells_by_type"]
    else:
        modules = report.get("modules", {})
        top = modules.get("\\reduced_vgg_inference", modules.get("reduced_vgg_inference"))
        if top is None:
            raise ValueError("Missing reduced_vgg_inference in Yosys JSON")
        cells = top["num_cells_by_type"]
        if any(name in modules or "\\" + name in modules for name in cells):
            raise ValueError("Hierarchical report requires aggregate design statistics")
    counts = {
        "LUT": sum(n for kind, n in cells.items() if kind in {f"LUT{i}" for i in range(1, 7)}),
        "BRAM18 equivalent": sum(n * (2 if kind.startswith("RAMB36") else 1)
                                 for kind, n in cells.items()
                                 if kind.startswith(("RAMB18", "RAMB36"))),
        "DSP": sum(n for kind, n in cells.items() if kind.startswith("DSP48")),
        "FF": sum(n for kind, n in cells.items() if kind.startswith(("FDRE", "FDSE", "FDCE", "FDPE"))),
    }
    budgets = {"LUT": 53200, "BRAM18 equivalent": 280, "DSP": 220, "FF": 106400}
    for kind, used in counts.items():
        print(f"{kind}: {used}/{budgets[kind]} ({used / budgets[kind] * 100:.1f}%)")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", default="yosys_stats.json")
    parse_report(parser.parse_args().report)
