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


def rect_float(rect: ET.Element, attr: str, default: float = 0.0) -> float:
    value = rect.get(attr)
    return float(value) if value is not None else default


def has_class(element: ET.Element, class_name: str) -> bool:
    return class_name in element.get("class", "").split()


def remove_existing_artifacts(svg: ET.Element) -> None:
    artifact_ids = {
        "crt-snake-defs",
        "crt-snake-style",
        "crt-snake-surface",
        "crt-progress-track",
        "crt-progress-label",
    }
    for child in list(svg):
        if child.get("id") in artifact_ids:
            svg.remove(child)


def find_rects(svg: ET.Element, class_name: str) -> list[ET.Element]:
    return [
        rect
        for rect in svg.findall(svg_tag("rect"))
        if has_class(rect, class_name)
    ]


def format_number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def compact_progress_bar(
    svg: ET.Element,
    progress_y: float,
    progress_height: float,
) -> tuple[float, float, float, float]:
    progress_rects = find_rects(svg, "u")
    if not progress_rects:
        raise ValueError("Could not find snake progress bar rectangles")

    current_y = min(rect_float(rect, "y") for rect in progress_rects)
    y_delta = progress_y - current_y
    min_x = min(rect_float(rect, "x") for rect in progress_rects)
    max_x = max(rect_float(rect, "x") + rect_float(rect, "width") for rect in progress_rects)

    for rect in progress_rects:
        rect.set("y", format_number(rect_float(rect, "y") + y_delta))
        rect.set("height", format_number(progress_height))
        rect.set("rx", "1")
        rect.set("ry", "1")

    return min_x, progress_y, max_x - min_x, progress_height


def inject_crt_surface(
    svg: ET.Element,
    surface_x: float,
    surface_y: float,
    surface_width: float,
    surface_height: float,
    progress_bounds: tuple[float, float, float, float],
) -> None:
    defs = ET.Element(svg_tag("defs"), {"id": "crt-snake-defs"})

    gradient = ET.SubElement(
        defs,
        svg_tag("linearGradient"),
        {"id": "crt-snake-bg", "x1": "0%", "y1": "0%", "x2": "0%", "y2": "100%"},
    )
    ET.SubElement(gradient, svg_tag("stop"), {"offset": "0%", "stop-color": "#071207"})
    ET.SubElement(gradient, svg_tag("stop"), {"offset": "56%", "stop-color": "#050905"})
    ET.SubElement(gradient, svg_tag("stop"), {"offset": "100%", "stop-color": "#030603"})

    pattern = ET.SubElement(
        defs,
        svg_tag("pattern"),
        {
            "id": "crt-scanlines",
            "width": "1",
            "height": "4",
            "patternUnits": "userSpaceOnUse",
        },
    )
    ET.SubElement(
        pattern,
        svg_tag("rect"),
        {"width": "1", "height": "1", "fill": "#D8FFC4", "opacity": "0.045"},
    )
    ET.SubElement(
        pattern,
        svg_tag("rect"),
        {"y": "2", "width": "1", "height": "1", "fill": "#000000", "opacity": "0.18"},
    )

    glow = ET.SubElement(
        defs,
        svg_tag("filter"),
        {"id": "crt-label-glow", "x": "-20%", "y": "-60%", "width": "140%", "height": "220%"},
    )
    ET.SubElement(glow, svg_tag("feGaussianBlur"), {"stdDeviation": "1.8", "result": "blur"})
    merge = ET.SubElement(glow, svg_tag("feMerge"))
    ET.SubElement(merge, svg_tag("feMergeNode"), {"in": "blur"})
    ET.SubElement(merge, svg_tag("feMergeNode"), {"in": "SourceGraphic"})

    clip = ET.SubElement(defs, svg_tag("clipPath"), {"id": "crt-snake-clip"})
    ET.SubElement(
        clip,
        svg_tag("rect"),
        {
            "x": format_number(surface_x),
            "y": format_number(surface_y),
            "width": format_number(surface_width),
            "height": format_number(surface_height),
            "rx": "18",
            "ry": "18",
        },
    )

    style = ET.Element(
        svg_tag("style"),
        {"id": "crt-snake-style"},
    )
    style.text = (
        ":root{--cb:#68FF7E3D;--cs:#D8FFC4;--ce:#050905;--c0:#050905;"
        "--c1:#153B1A;--c2:#22592A;--c3:#5DA153;--c4:#B6FF9A}"
        ".c{shape-rendering:crispEdges;stroke-width:1.15px}"
        ".u{shape-rendering:crispEdges;opacity:.86}"
        ".s{filter:url(#crt-label-glow)}"
    )

    surface = ET.Element(svg_tag("g"), {"id": "crt-snake-surface"})
    ET.SubElement(
        surface,
        svg_tag("rect"),
        {
            "x": format_number(surface_x),
            "y": format_number(surface_y),
            "width": format_number(surface_width),
            "height": format_number(surface_height),
            "rx": "18",
            "ry": "18",
            "fill": "url(#crt-snake-bg)",
        },
    )
    ET.SubElement(
        surface,
        svg_tag("rect"),
        {
            "x": format_number(surface_x + 4),
            "y": format_number(surface_y + 4),
            "width": format_number(surface_width - 8),
            "height": format_number(surface_height - 8),
            "rx": "15",
            "ry": "15",
            "fill": "none",
            "stroke": "#68FF7E",
            "stroke-opacity": "0.18",
            "stroke-width": "2",
        },
    )
    ET.SubElement(
        surface,
        svg_tag("rect"),
        {
            "x": format_number(surface_x),
            "y": format_number(surface_y),
            "width": format_number(surface_width),
            "height": format_number(surface_height),
            "rx": "18",
            "ry": "18",
            "fill": "url(#crt-scanlines)",
            "clip-path": "url(#crt-snake-clip)",
            "opacity": "0.6",
        },
    )

    progress_x, progress_y, progress_width, progress_height = progress_bounds
    track = ET.Element(
        svg_tag("rect"),
        {
            "id": "crt-progress-track",
            "x": format_number(progress_x),
            "y": format_number(progress_y),
            "width": format_number(progress_width),
            "height": format_number(progress_height),
            "rx": "1",
            "ry": "1",
            "fill": "#071307",
            "stroke": "#68FF7E",
            "stroke-opacity": "0.16",
            "stroke-width": "1",
        },
    )

    insert_at = 0
    if len(svg) and svg[0].tag == svg_tag("desc"):
        insert_at = 1
    svg.insert(insert_at, defs)
    insert_at += 1
    while insert_at < len(svg) and svg[insert_at].tag == svg_tag("style"):
        insert_at += 1
    svg.insert(insert_at, style)
    svg.insert(insert_at + 1, surface)
    svg.insert(insert_at + 2, track)


def inject_label(path: Path, percentage: str) -> None:
    tree = ET.parse(path)
    svg = tree.getroot()
    min_x, min_y, width, height = parse_view_box(svg)

    remove_existing_artifacts(svg)

    surface_y = -18
    surface_height = 170
    progress_bounds = compact_progress_bar(svg, progress_y=122, progress_height=10)
    progress_x, progress_y, progress_width, progress_height = progress_bounds

    min_y = surface_y
    height = surface_height
    svg.set("viewBox", f"{format_number(min_x)} {format_number(min_y)} {format_number(width)} {format_number(height)}")
    svg.set("width", format_number(width))
    svg.set("height", format_number(height))

    inject_crt_surface(svg, min_x, min_y, width, height, progress_bounds)

    box_width = 56
    box_height = 18
    box_x = progress_x + progress_width - box_width - 2
    box_y = progress_y - 4
    text_x = box_x + box_width / 2
    text_y = box_y + 12.5

    group = ET.Element(
        svg_tag("g"),
        {
            "id": "crt-progress-label",
            "font-family": "Consolas, 'Courier New', monospace",
            "font-size": "10",
            "font-weight": "700",
            "letter-spacing": "0.6",
            "text-anchor": "middle",
            "shape-rendering": "crispEdges",
            "filter": "url(#crt-label-glow)",
        },
    )
    ET.SubElement(
        group,
        svg_tag("rect"),
        {
            "x": format_number(box_x),
            "y": format_number(box_y),
            "width": format_number(box_width),
            "height": format_number(box_height),
            "rx": "2",
            "ry": "2",
            "fill": "#050905",
            "stroke": "#68FF7E",
            "stroke-opacity": "0.44",
        },
    )
    ET.SubElement(
        group,
        svg_tag("path"),
        {
            "d": (
                f"M {format_number(box_x + 4)} {format_number(box_y + box_height - 3)} "
                f"H {format_number(box_x + box_width - 4)}"
            ),
            "stroke": "#68FF7E",
            "stroke-opacity": "0.28",
            "stroke-width": "1",
        },
    )
    ET.SubElement(
        group,
        svg_tag("text"),
        {
            "x": format_number(text_x + 1),
            "y": format_number(text_y + 1),
            "fill": "#68FF7E",
            "opacity": "0.32",
        },
    ).text = percentage
    ET.SubElement(
        group,
        svg_tag("text"),
        {
            "x": format_number(text_x),
            "y": format_number(text_y),
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
