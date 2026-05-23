"""
Add a CRT-style percentage label to Platane/snk SVG outputs.

The generated files live on the workflow's `dist/` directory and are pushed to
the `output` branch, so this script intentionally edits generated SVGs in CI
instead of committing those SVGs to main.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def parse_view_box(svg: ET.Element) -> tuple[float, float, float, float]:
    view_box = svg.get("viewBox")
    if not view_box:
        raise ValueError("SVG is missing a viewBox")

    parts = [float(part) for part in view_box.replace(",", " ").split()]
    if len(parts) != 4:
        raise ValueError(f"Unsupported viewBox format: {view_box}")
    return parts[0], parts[1], parts[2], parts[3]


def remove_existing_label(svg: ET.Element) -> None:
    for child in list(svg):
        if child.get("id") == "crt-progress-label":
            svg.remove(child)


def inject_label(path: Path, percentage: str) -> None:
    tree = ET.parse(path)
    svg = tree.getroot()
    min_x, min_y, width, height = parse_view_box(svg)

    label_x = min_x + width - 12
    label_y = min_y + height - 11
    box_width = 64
    box_height = 24

    remove_existing_label(svg)

    group = ET.Element(
        svg_tag("g"),
        {
            "id": "crt-progress-label",
            "font-family": "Consolas, 'Courier New', monospace",
            "font-size": "10",
            "font-weight": "700",
            "letter-spacing": "1",
            "text-anchor": "end",
            "shape-rendering": "crispEdges",
        },
    )
    ET.SubElement(
        group,
        svg_tag("rect"),
        {
            "x": f"{label_x - box_width:.1f}",
            "y": f"{label_y - box_height + 6:.1f}",
            "width": f"{box_width:.1f}",
            "height": f"{box_height:.1f}",
            "rx": "3",
            "ry": "3",
            "fill": "#050905",
            "stroke": "#68FF7E",
            "stroke-opacity": "0.38",
        },
    )
    ET.SubElement(
        group,
        svg_tag("text"),
        {
            "x": f"{label_x:.1f}",
            "y": f"{label_y:.1f}",
            "fill": "#68FF7E",
            "opacity": "0.45",
        },
    ).text = percentage
    ET.SubElement(
        group,
        svg_tag("text"),
        {
            "x": f"{label_x - 1:.1f}",
            "y": f"{label_y - 1:.1f}",
            "fill": "#D8FFC4",
        },
    ).text = percentage

    svg.append(group)
    tree.write(path, encoding="unicode", xml_declaration=False)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", nargs="+", type=Path)
    parser.add_argument("--percentage", default="100%")
    args = parser.parse_args()

    for svg_path in args.svg:
        inject_label(svg_path, args.percentage)
        print(f"Added CRT progress label to {svg_path}")


if __name__ == "__main__":
    main()
