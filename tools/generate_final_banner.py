"""
Generate a retro CRT terminal style GIF banner using high-quality ASCII art
from ascii-image-converter as base, then add neofetch panel and CRT effects.
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random

# ============ CONFIGURATION ============
CONFIG = {
    # Personal info (neofetch style)
    "name": "NJX",
    "age": "22", 
    "work": "BIT · CS",
    "os": "macOS",
    "editor": "VS Code",
    "languages": ["Python, Go,", "TypeScript, Rust,", "C++, Bash"],
    "skills": ["Vision, LLM,", "Agents, Infra"],
    "prompt": "njx@mbp$",
    
    # Animation
    "frames": 8,
    "frame_duration": 200,
}


def load_font(size):
    """Load a monospace font."""
    font_paths = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/lucon.ttf", 
        "C:/Windows/Fonts/cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()


def add_scanlines(img, intensity=0.15, line_gap=2):
    """Add CRT scanline effect."""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    for y in range(0, height, line_gap):
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, int(255 * intensity)), width=1)
    
    return img


def add_glow(img, radius=2):
    """Add phosphor glow effect."""
    glow = img.filter(ImageFilter.GaussianBlur(radius))
    return Image.blend(img, glow, 0.3)


def draw_info_panel(draw, x, y, font, small_font):
    """Draw neofetch-style info panel."""
    cfg = CONFIG
    
    # Colors
    green = (50, 255, 50)
    bright_green = (150, 255, 150)
    white = (255, 255, 255)
    cyan = (0, 255, 255)
    yellow = (255, 255, 100)
    
    line_height = 28
    label_width = 95
    
    # Header with name
    draw.text((x, y), f"╭─ {cfg['name']} ", font=font, fill=bright_green)
    y += line_height + 5
    
    # Separator
    draw.text((x, y), "├" + "─" * 20, font=font, fill=green)
    y += line_height
    
    # Info lines
    info_items = [
        ("name", cfg["name"]),
        ("age", cfg["age"]),
        ("work", cfg["work"]),
        ("os", cfg["os"]),
        ("editor", cfg["editor"]),
    ]
    
    for label, value in info_items:
        draw.text((x, y), f"│ {label}", font=small_font, fill=cyan)
        draw.text((x + label_width, y), value, font=small_font, fill=white)
        y += line_height
    
    # Languages
    draw.text((x, y), f"│ lang", font=small_font, fill=cyan)
    draw.text((x + label_width, y), cfg["languages"][0], font=small_font, fill=yellow)
    y += line_height
    
    for lang in cfg["languages"][1:]:
        draw.text((x, y), "│", font=small_font, fill=green)
        draw.text((x + label_width, y), lang, font=small_font, fill=yellow)
        y += line_height
    
    # Skills
    draw.text((x, y), f"│ skills", font=small_font, fill=cyan)
    draw.text((x + label_width, y), cfg["skills"][0], font=small_font, fill=white)
    y += line_height
    
    for skill in cfg["skills"][1:]:
        draw.text((x, y), "│", font=small_font, fill=green)
        draw.text((x + label_width, y), skill, font=small_font, fill=white)
        y += line_height
    
    # Footer
    y += 5
    draw.text((x, y), "╰" + "─" * 20, font=font, fill=green)
    y += line_height + 10
    
    # Prompt with cursor
    draw.text((x, y), cfg["prompt"], font=font, fill=green)
    
    return y


def create_frame(ascii_img, frame_num, total_frames):
    """Create a single animation frame."""
    # Get ASCII art dimensions
    ascii_w, ascii_h = ascii_img.size
    
    # Calculate total canvas size (ASCII + info panel)
    panel_width = 320
    total_width = ascii_w + panel_width + 40  # padding
    total_height = max(ascii_h + 40, 600)
    
    # Create frame with dark background
    frame = Image.new('RGBA', (total_width, total_height), (10, 15, 10, 255))
    
    # Paste ASCII art on left
    frame.paste(ascii_img, (20, 20))
    
    # Draw info panel on right
    draw = ImageDraw.Draw(frame)
    font = load_font(22)
    small_font = load_font(20)
    
    panel_x = ascii_w + 50
    panel_y = 60
    final_y = draw_info_panel(draw, panel_x, panel_y, font, small_font)
    
    # Blinking cursor animation
    cursor_visible = frame_num % 2 == 0
    if cursor_visible:
        cursor_x = panel_x + len(CONFIG["prompt"]) * 13 + 5
        cursor_y = final_y
        draw.rectangle([cursor_x, cursor_y, cursor_x + 12, cursor_y + 22], fill=(50, 255, 50))
    
    # Add CRT effects
    frame = frame.convert('RGB')
    
    # Slight random noise for CRT flicker
    if random.random() > 0.7:
        enhancer = ImageEnhance.Brightness(frame)
        frame = enhancer.enhance(random.uniform(0.97, 1.03))
    
    # Add scanlines with varying intensity
    frame = frame.convert('RGBA')
    frame = add_scanlines(frame, intensity=0.12 + random.uniform(-0.02, 0.02))
    
    # Add subtle glow
    frame = frame.convert('RGB')
    frame = add_glow(frame, radius=1)
    
    return frame


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Load the ASCII art generated by ascii-image-converter
    ascii_path = os.path.join(project_dir, "images", "ascii_base.png")
    
    if not os.path.exists(ascii_path):
        print(f"Error: ASCII art not found at {ascii_path}")
        print("Please run: ./tools/ascii-image-converter.exe images/girl.jpg -c -W 65 -s . --font-color 50,255,50 --save-bg 0,0,0,100 --only-save")
        return
    
    ascii_img = Image.open(ascii_path).convert('RGBA')
    print(f"Loaded ASCII art: {ascii_img.size}")
    
    # Generate frames
    frames = []
    total_frames = CONFIG["frames"]
    
    print(f"Generating {total_frames} frames...")
    for i in range(total_frames):
        print(f"  Frame {i+1}/{total_frames}")
        frame = create_frame(ascii_img, i, total_frames)
        frames.append(frame)
    
    # Save as GIF
    output_path = os.path.join(project_dir, "images", "crt_banner.gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=CONFIG["frame_duration"],
        loop=0,
        optimize=False
    )
    
    print(f"\nSaved: {output_path}")
    
    # Also save a static PNG version
    static_path = os.path.join(project_dir, "images", "crt_banner.png")
    frames[0].save(static_path)
    print(f"Saved static: {static_path}")


if __name__ == "__main__":
    main()
