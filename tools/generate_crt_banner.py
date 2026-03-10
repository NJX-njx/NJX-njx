"""
Generate the current CRT README banner into `assets/banners/`.

Portrait sources live under `images/portraits/`, while the output banner is
kept under `assets/banners/`.
"""

from collections import deque
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


CONFIG = {
    "name": "NJX",
    "age": "19",
    "work": "BIT · CS",
    "os": "Windows / macOS",
    "editor": ["Cursor / VS Code", "Opencode / Trae"],
    "languages": ["Python, CUDA,", "C/C++, JavaScript,", "TypeScript, Bash"],
    "skills": [
        "LLM fine-tuning,",
        "Prompt/context eng,",
        "Distributed training,",
        "LLM evaluation",
    ],
    "recent": ["Watching The Young", "Brewmaster's Adventure"],
    "prompt": "njx@mbp$",
    "width": 1200,
    "height": 700,
    "frames": 6,
    "frame_duration": 220,
    "portrait_source": "images/portraits/portrait-reference.jpg",
}


def load_font(size: int):
    font_paths = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/lucon.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for path in font_paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_glow_text(image: Image.Image, position: tuple[int, int], text: str, font, fill, glow, blur_radius: int = 6) -> None:
    glow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.text(position, text, font=font, fill=glow)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(blur_radius))
    image.alpha_composite(glow_layer)
    ImageDraw.Draw(image).text(position, text, font=font, fill=fill)


def left_screen_color(y: int, frame_num: int) -> tuple[int, int, int]:
    phase = (y + frame_num) % 4
    if phase == 0:
        return (190, 255, 156)
    if phase == 1:
        return (173, 246, 141)
    if phase == 2:
        return (161, 236, 128)
    return (148, 226, 118)


def right_screen_color(y: int, frame_num: int) -> tuple[int, int, int]:
    base = 12 + ((y + frame_num) % 5) * 2
    return (base, 22 + base, base)


def draw_screen_background(draw: ImageDraw.ImageDraw, cfg: dict, frame_num: int) -> None:
    left_x, top_y, left_w, screen_h = 34, 34, 664, 632
    right_x, right_w = 706, 460

    for y in range(top_y, top_y + screen_h):
        draw.line((left_x, y, left_x + left_w, y), fill=left_screen_color(y, frame_num))
        draw.line((right_x, y, right_x + right_w, y), fill=right_screen_color(y, frame_num))

    # subtle phosphor dots on the bright panel
    for y in range(top_y + 6, top_y + screen_h, 18):
        for x in range(left_x + 6, left_x + left_w, 24):
            draw.point((x, y), fill=(124, 206, 101))


def draw_bar_block(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, fill, accent) -> None:
    draw.rounded_rectangle((x, y, x + width, y + height), radius=2, fill=fill)
    draw.line((x + 2, y + 2, x + width - 2, y + 2), fill=accent, width=1)
    draw.line((x + 2, y + height - 3, x + width - 2, y + height - 3), fill=accent, width=1)
    for offset in (5, 10, width - 12, width - 7):
        if 0 < offset < width:
            draw.point((x + offset, y + height // 2), fill=accent)


def average_color(samples: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    count = max(len(samples), 1)
    red = sum(pixel[0] for pixel in samples) / count
    green = sum(pixel[1] for pixel in samples) / count
    blue = sum(pixel[2] for pixel in samples) / count
    return red, green, blue


def color_distance(pixel: tuple[int, int, int], ref: tuple[float, float, float]) -> float:
    return abs(pixel[0] - ref[0]) + abs(pixel[1] - ref[1]) + abs(pixel[2] - ref[2])


def build_subject_mask(source: Image.Image) -> Image.Image:
    rgb = source.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    corner_size = max(min(width, height) // 8, 24)
    background_samples = []

    for x in range(corner_size):
        for y in range(corner_size):
            background_samples.append(pixels[x, y])
            background_samples.append(pixels[width - 1 - x, y])
            background_samples.append(pixels[x, height - 1 - y])
            background_samples.append(pixels[width - 1 - x, height - 1 - y])

    background_color = average_color(background_samples)
    threshold = 78
    background = [[False for _ in range(width)] for _ in range(height)]
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))

    while queue:
        x, y = queue.popleft()
        if x < 0 or x >= width or y < 0 or y >= height or background[y][x]:
            continue
        pixel = pixels[x, y]
        if color_distance(pixel, background_color) > threshold:
            continue
        background[y][x] = True
        queue.append((x + 1, y))
        queue.append((x - 1, y))
        queue.append((x, y + 1))
        queue.append((x, y - 1))

    mask = Image.new("L", (width, height), 0)
    mask_pixels = mask.load()
    for y in range(height):
        for x in range(width):
            if background[y][x]:
                continue
            mask_pixels[x, y] = 255

    # Keep the portrait silhouette soft enough to survive CRT blur.
    mask = mask.filter(ImageFilter.MedianFilter(size=5))
    mask = mask.filter(ImageFilter.GaussianBlur(1.1))
    return mask.point(lambda value: 255 if value > 32 else 0)


def crop_focus_portrait(source: Image.Image) -> Image.Image:
    width, height = source.size
    crop_box = (
        int(width * 0.03),
        int(height * 0.00),
        int(width * 0.97),
        int(height * 0.97),
    )
    return source.crop(crop_box)


def build_focus_mask(size: tuple[int, int]) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    draw.ellipse((width * 0.16, height * 0.01, width * 0.80, height * 0.56), fill=255)
    draw.ellipse((width * 0.03, height * 0.16, width * 0.31, height * 0.96), fill=220)
    draw.ellipse((width * 0.63, height * 0.18, width * 0.88, height * 0.90), fill=205)
    draw.rounded_rectangle((width * 0.18, height * 0.47, width * 0.90, height * 0.86), radius=int(width * 0.08), fill=238)
    draw.ellipse((width * 0.00, height * 0.47, width * 0.34, height * 0.82), fill=244)
    draw.rectangle((width * 0.28, height * 0.50, width * 0.56, height * 0.84), fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(12))
    return mask.point(lambda value: 255 if value > 36 else 0)


def crop_subject(source: Image.Image, mask: Image.Image) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return source.convert("RGBA")

    left, top, right, bottom = bbox
    pad_x = max((right - left) // 10, 26)
    pad_y = max((bottom - top) // 10, 24)
    crop_box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(source.width, right + pad_x),
        min(source.height, bottom + pad_y),
    )

    portrait = Image.new("RGBA", source.size, (0, 0, 0, 0))
    portrait.paste(source.convert("RGBA"), (0, 0), mask)
    return portrait.crop(crop_box)


def palette_for_luminance(luminance: float, edge_strength: float) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if luminance < 58:
        fill, accent = (42, 93, 45), (26, 61, 29)
    elif luminance < 92:
        fill, accent = (61, 122, 59), (39, 80, 38)
    elif luminance < 126:
        fill, accent = (93, 161, 83), (58, 112, 54)
    elif luminance < 164:
        fill, accent = (138, 199, 112), (93, 146, 78)
    elif luminance < 204:
        fill, accent = (200, 232, 176), (126, 172, 104)
    else:
        fill, accent = (242, 246, 225), (168, 188, 145)

    if edge_strength > 66:
        accent = tuple(max(channel - 22, 0) for channel in accent)
    elif edge_strength < 24:
        accent = tuple(min(channel + 10, 255) for channel in accent)
    return fill, accent


def preprocess_portrait(portrait: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    rgb = portrait.convert("RGB")
    grayscale = ImageOps.grayscale(rgb)
    grayscale = ImageOps.autocontrast(grayscale, cutoff=2)
    grayscale = ImageEnhance.Contrast(grayscale).enhance(1.35)
    grayscale = ImageEnhance.Sharpness(grayscale).enhance(1.2)

    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(1.8)
    softened = grayscale.filter(ImageFilter.GaussianBlur(0.4))
    return portrait.getchannel("A"), softened, edges


def build_portrait_blocks(project_root: Path) -> list[tuple[int, int, tuple[int, int, int], tuple[int, int, int]]]:
    source_path = project_root / CONFIG["portrait_source"]
    print(f"debugging: loading portrait source from {source_path}")
    source = Image.open(source_path).convert("RGB")
    focused_source = crop_focus_portrait(source)
    auto_mask = build_subject_mask(focused_source)
    focus_mask = build_focus_mask(focused_source.size)
    combined_mask = ImageChops.multiply(auto_mask, focus_mask)
    portrait = crop_subject(focused_source, combined_mask)

    grid_height = 38
    grid_width = max(18, round(portrait.width / portrait.height * grid_height))
    resized = portrait.resize((grid_width, grid_height), Image.Resampling.LANCZOS)
    alpha, grayscale, edges = preprocess_portrait(resized)

    blocks = []
    for row in range(grid_height):
        for col in range(grid_width):
            alpha_value = alpha.getpixel((col, row))
            if alpha_value < 56:
                continue
            luminance = grayscale.getpixel((col, row))
            edge_strength = edges.getpixel((col, row))
            fill, accent = palette_for_luminance(luminance, edge_strength)
            blocks.append((col, row, fill, accent))

    print(f"debugging: extracted {len(blocks)} portrait blocks from source image")
    return blocks


def fit_cover(image: Image.Image, size: tuple[int, int], centering: tuple[float, float]) -> Image.Image:
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=centering)


def prepare_portrait_panel(project_root: Path, cfg: dict) -> Image.Image:
    source_path = project_root / cfg["portrait_source"]
    print(f"debugging: loading direct portrait panel from {source_path}")
    source = Image.open(source_path).convert("RGB")
    focused_source = crop_focus_portrait(source)

    # Keep the original illustration, but push the room background back.
    subject_mask = build_subject_mask(focused_source)
    focus_mask = build_focus_mask(focused_source.size)
    emphasis_mask = ImageChops.lighter(subject_mask, focus_mask)

    softened_background = ImageEnhance.Color(focused_source).enhance(0.86)
    softened_background = ImageEnhance.Brightness(softened_background).enhance(0.84)
    softened_background = softened_background.filter(ImageFilter.GaussianBlur(5))

    portrait_layer = Image.new("RGBA", focused_source.size, (0, 0, 0, 0))
    portrait_layer.paste(focused_source.convert("RGBA"), (0, 0), emphasis_mask)

    composed = softened_background.convert("RGBA")
    composed.alpha_composite(portrait_layer)
    panel = Image.new("RGBA", (664, 632), (0, 0, 0, 0))
    contained = ImageOps.contain(composed, (628, 620), method=Image.Resampling.LANCZOS)
    offset_x = (panel.width - contained.width) // 2 - 10
    offset_y = (panel.height - contained.height) // 2 + 2
    panel.alpha_composite(contained, dest=(offset_x, offset_y))
    print("debugging: prepared direct portrait panel")
    return panel


def stylize_portrait_panel(portrait_panel: Image.Image, frame_num: int) -> Image.Image:
    panel_rgb = portrait_panel.convert("RGB")
    panel_rgb = ImageEnhance.Color(panel_rgb).enhance(0.92)
    panel_rgb = ImageEnhance.Contrast(panel_rgb).enhance(1.08)
    panel_rgb = ImageEnhance.Sharpness(panel_rgb).enhance(1.12)

    phosphor = Image.new("RGB", panel_rgb.size, (152, 232, 143))
    monochrome = ImageOps.grayscale(panel_rgb).convert("RGB")
    panel_rgb = Image.blend(panel_rgb, monochrome, 0.06)
    panel_rgb = Image.blend(panel_rgb, phosphor, 0.08)

    glow = panel_rgb.filter(ImageFilter.GaussianBlur(1.3))
    panel_rgb = Image.blend(panel_rgb, glow, 0.07)

    panel = panel_rgb.convert("RGBA")
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    width, height = panel.size

    for y in range(0, height, 4):
        bright_alpha = 6 + ((frame_num + y) % 3) * 3
        overlay_draw.line((0, y, width, y), fill=(170, 255, 188, bright_alpha))
        dark_y = min(y + 2, height - 1)
        overlay_draw.line((0, dark_y, width, dark_y), fill=(4, 18, 4, 10))

    panel.alpha_composite(overlay)

    panel_shadow = Image.new("L", panel.size, 0)
    shadow_draw = ImageDraw.Draw(panel_shadow)
    shadow_draw.ellipse((-60, -24, width + 60, height + 24), fill=230)
    panel_shadow = panel_shadow.filter(ImageFilter.GaussianBlur(28))
    dark_panel = Image.new("RGBA", panel.size, (8, 14, 8, 255))
    panel = Image.composite(panel, dark_panel, panel_shadow)
    return panel


def draw_portrait(image: Image.Image, portrait_panel: Image.Image, frame_num: int) -> None:
    styled_panel = stylize_portrait_panel(portrait_panel, frame_num)
    image.alpha_composite(styled_panel, dest=(34, 34))


def draw_info_panel(image: Image.Image, cfg: dict, frame_num: int) -> None:
    font_main = load_font(18)
    font_value = load_font(18)
    font_prompt = load_font(18)
    green = (149, 255, 121, 255)
    bright = (216, 255, 196, 255)
    glow = (104, 255, 109, 150)

    x = 742
    y = 70
    line_gap = 40
    value_x = x + 118

    items = [
        ("Name:", cfg["name"]),
        ("Age:", cfg["age"]),
        ("Work:", cfg["work"]),
        ("OS:", cfg["os"]),
    ]

    for label, value in items:
        draw_glow_text(image, (x, y), label, font_main, green, glow, blur_radius=4)
        draw_glow_text(image, (value_x, y), value, font_value, bright, glow, blur_radius=4)
        y += line_gap

    draw_glow_text(image, (x, y), "Editor:", font_main, green, glow, blur_radius=4)
    draw_glow_text(image, (value_x, y), cfg["editor"][0], font_value, bright, glow, blur_radius=4)
    y += 34
    for line in cfg["editor"][1:]:
        draw_glow_text(image, (value_x, y), line, font_value, bright, glow, blur_radius=4)
        y += 34

    y += 8
    draw_glow_text(image, (x, y), "Languages:", font_main, green, glow, blur_radius=4)
    draw_glow_text(image, (value_x, y), cfg["languages"][0], font_value, bright, glow, blur_radius=4)
    y += 34
    for line in cfg["languages"][1:]:
        draw_glow_text(image, (value_x, y), line, font_value, bright, glow, blur_radius=4)
        y += 34

    y += 8
    draw_glow_text(image, (x, y), "Skills:", font_main, green, glow, blur_radius=4)
    draw_glow_text(image, (value_x, y), cfg["skills"][0], font_value, bright, glow, blur_radius=4)
    y += 34
    for line in cfg["skills"][1:]:
        draw_glow_text(image, (value_x, y), line, font_value, bright, glow, blur_radius=4)
        y += 34

    y += 10
    draw_glow_text(image, (x, y), "Recent:", font_main, green, glow, blur_radius=4)
    draw_glow_text(image, (value_x, y), cfg["recent"][0], font_value, bright, glow, blur_radius=4)
    y += 34
    for line in cfg["recent"][1:]:
        draw_glow_text(image, (value_x, y), line, font_value, bright, glow, blur_radius=4)
        y += 34

    prompt_y = 628
    draw_glow_text(image, (48, prompt_y), cfg["prompt"], font_prompt, bright, glow, blur_radius=4)
    if frame_num % 2 == 0:
        cursor_x = int(48 + ImageDraw.Draw(image).textlength(cfg["prompt"], font=font_prompt) + 8)
        cursor_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        cursor_draw = ImageDraw.Draw(cursor_layer)
        cursor_draw.rectangle((cursor_x, prompt_y + 2, cursor_x + 12, prompt_y + 22), fill=(222, 255, 215, 255))
        cursor_layer = cursor_layer.filter(ImageFilter.GaussianBlur(3))
        image.alpha_composite(cursor_layer)
        ImageDraw.Draw(image).rectangle((cursor_x, prompt_y + 2, cursor_x + 12, prompt_y + 22), fill=(222, 255, 215, 255))


def add_monitor_effects(image: Image.Image, frame_num: int) -> Image.Image:
    width, height = image.size
    base = image.convert("RGB")
    flicker = 0.988 + (frame_num % 3) * 0.014
    base = ImageEnhance.Brightness(base).enhance(flicker)

    bloom = base.filter(ImageFilter.GaussianBlur(2.2))
    base = Image.blend(base, bloom, 0.24)

    vignette = Image.new("L", (width, height), 0)
    vignette_draw = ImageDraw.Draw(vignette)
    vignette_draw.ellipse((-120, -70, width + 120, height + 70), fill=230)
    vignette = vignette.filter(ImageFilter.GaussianBlur(46))

    dark = Image.new("RGB", (width, height), (6, 11, 6))
    base = Image.composite(base, dark, vignette)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle((10, 10, width - 10, height - 10), radius=30, outline=(13, 22, 13), width=24)
    overlay_draw.rounded_rectangle((20, 20, width - 20, height - 20), radius=28, outline=(6, 12, 6), width=12)
    overlay_draw.line((698, 36, 698, 664), fill=(18, 42, 18, 180), width=4)

    glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.rounded_rectangle((28, 28, 1172, 672), radius=26, outline=(104, 255, 126, 70), width=6)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(12))

    final = base.convert("RGBA")
    final.alpha_composite(glow_layer)
    final.alpha_composite(overlay)
    return final.convert("RGB")


def create_frame(cfg: dict, frame_num: int, portrait_panel: Image.Image) -> Image.Image:
    image = Image.new("RGBA", (cfg["width"], cfg["height"]), (5, 9, 5, 255))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((16, 16, 1184, 684), radius=34, fill=(7, 12, 7))
    draw_screen_background(draw, cfg, frame_num)
    draw_portrait(image, portrait_panel, frame_num)
    draw_info_panel(image, cfg, frame_num)
    return add_monitor_effects(image, frame_num)


def generate_crt_gif(output_path: Path, cfg: dict) -> None:
    print("debugging: generating direct-portrait crt banner")
    portrait_panel = prepare_portrait_panel(Path(__file__).resolve().parent.parent, cfg)
    frames = []
    for index in range(cfg["frames"]):
        print(f"debugging: rendering direct portrait frame {index + 1}/{cfg['frames']}")
        frames.append(create_frame(cfg, index, portrait_panel))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=cfg["frame_duration"],
        loop=0,
        optimize=False,
    )
    print(f"debugging: saved direct portrait banner to {output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_gif = project_root / "assets" / "banners" / "crt-banner.gif"
    generate_crt_gif(output_gif, CONFIG)


if __name__ == "__main__":
    main()
