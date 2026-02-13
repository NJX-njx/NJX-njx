"""
Generate a retro CRT terminal style GIF banner (like yetone's profile)
Features:
- ASCII art from image (left side)
- Neofetch-style info panel (right side)
- CRT scanlines & phosphor glow effect
- Blinking cursor animation
- Terminal prompt at bottom
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import math

# ============ CONFIGURATION ============
CONFIG = {
    # Personal info (neofetch style)
    "name": "NJX",
    "affiliation": "BIT · CS",
    "focus": "AI Engineer",
    "os": "Linux/macOS",
    "editor": "VS Code",
    "languages": "Python, C++,\n            TypeScript, Go",
    "skills": "Vision, LLM,\n         Agents, Infra",
    
    # Terminal
    "prompt": "njx@dev$",
    
    # Colors (CRT green phosphor)
    "bg_color": (10, 20, 10),
    "text_color": (0, 255, 65),
    "dim_color": (0, 180, 45),
    "bright_color": (150, 255, 150),
    
    # Dimensions
    "width": 1000,
    "height": 500,
    "ascii_width": 50,  # characters for ASCII art
    
    # Animation
    "frames": 10,
    "frame_duration": 150,  # ms per frame
}

# ASCII characters from dark to light
ASCII_CHARS = " .:-=+*#%@"


def load_font(size):
    """Load a monospace font, fallback to default if not found."""
    font_paths = [
        "C:/Windows/Fonts/consola.ttf",  # Windows Consolas
        "C:/Windows/Fonts/cour.ttf",      # Windows Courier
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
        "/System/Library/Fonts/Menlo.ttc",  # macOS
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    return ImageFont.load_default()


def image_to_ascii(image_path, width=50):
    """Convert image to ASCII art string."""
    img = Image.open(image_path)
    
    # Calculate height to maintain aspect ratio (chars are ~2x taller than wide)
    aspect_ratio = img.height / img.width
    height = int(width * aspect_ratio * 0.5)
    
    # Resize and convert to grayscale
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img = img.convert("L")
    
    # Map pixels to ASCII characters
    pixels = list(img.getdata())
    ascii_str = ""
    for i, pixel in enumerate(pixels):
        # Map 0-255 to ASCII chars index
        char_idx = int(pixel / 255 * (len(ASCII_CHARS) - 1))
        ascii_str += ASCII_CHARS[char_idx]
        if (i + 1) % width == 0:
            ascii_str += "\n"
    
    return ascii_str.strip()


def create_scanlines(img, intensity=0.15):
    """Add CRT scanline effect."""
    draw = ImageDraw.Draw(img)
    for y in range(0, img.height, 2):
        draw.line([(0, y), (img.width, y)], fill=(0, 0, 0), width=1)
    
    # Reduce overall brightness for scanline effect
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1 - intensity * 0.3)


def create_crt_curve(img):
    """Add subtle CRT screen curvature effect (vignette)."""
    width, height = img.size
    
    # Create a vignette mask
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)
    
    # Radial gradient for vignette
    center_x, center_y = width // 2, height // 2
    max_dist = math.sqrt(center_x**2 + center_y**2)
    
    for y in range(height):
        for x in range(width):
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Stronger vignette at edges
            factor = 1 - (dist / max_dist) ** 2 * 0.4
            mask.putpixel((x, y), int(255 * max(0.6, factor)))
    
    # Apply vignette
    result = Image.new("RGB", img.size)
    result.paste(img)
    
    # Darken edges
    dark = Image.new("RGB", img.size, (0, 0, 0))
    result = Image.composite(result, dark, mask)
    
    return result


def add_glow(img, radius=2):
    """Add phosphor glow effect to text."""
    # Create a blurred copy
    glow = img.filter(ImageFilter.GaussianBlur(radius))
    
    # Blend original with glow
    return Image.blend(img, glow, 0.3)


def draw_color_blocks(draw, x, y, block_size=20):
    """Draw neofetch-style color blocks."""
    colors = [
        (40, 40, 40),     # Black
        (255, 85, 85),    # Red
        (85, 255, 85),    # Green
        (255, 255, 85),   # Yellow
        (85, 85, 255),    # Blue
        (255, 85, 255),   # Magenta
        (85, 255, 255),   # Cyan
        (255, 255, 255),  # White
    ]
    
    for i, color in enumerate(colors):
        draw.rectangle(
            [x + i * block_size, y, x + (i + 1) * block_size - 2, y + block_size - 2],
            fill=color
        )


def create_frame(ascii_art, config, frame_num, total_frames):
    """Create a single frame of the animation."""
    width = config["width"]
    height = config["height"]
    
    # Create base image
    img = Image.new("RGB", (width, height), config["bg_color"])
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    font_small = load_font(12)
    font_medium = load_font(14)
    font_large = load_font(16)
    
    # ===== Draw ASCII Art (Left Side) =====
    ascii_lines = ascii_art.split("\n")
    ascii_x = 30
    ascii_y = 40
    line_height = 10
    
    for i, line in enumerate(ascii_lines):
        # Slight color variation for depth
        color = config["text_color"]
        draw.text((ascii_x, ascii_y + i * line_height), line, font=font_small, fill=color)
    
    # ===== Draw Info Panel (Right Side) =====
    info_x = 550
    info_y = 80
    line_spacing = 28
    
    info_items = [
        ("Name:", config["name"]),
        ("Affiliation:", config["affiliation"]),
        ("Focus:", config["focus"]),
        ("OS:", config["os"]),
        ("Editor:", config["editor"]),
        ("Languages:", config["languages"]),
        ("Skills:", config["skills"]),
    ]
    
    current_y = info_y
    for label, value in info_items:
        # Label in dim color
        draw.text((info_x, current_y), label, font=font_medium, fill=config["dim_color"])
        
        # Value in bright color (handle multiline)
        value_lines = value.split("\n")
        label_width = draw.textlength(label, font=font_medium) + 10
        
        for j, vline in enumerate(value_lines):
            y_offset = current_y + j * 18
            draw.text((info_x + label_width, y_offset), vline, font=font_medium, fill=config["bright_color"])
        
        current_y += line_spacing + (len(value_lines) - 1) * 15
    
    # ===== Draw Color Blocks =====
    draw_color_blocks(draw, info_x, current_y + 20)
    
    # ===== Draw Terminal Prompt =====
    prompt_y = height - 50
    prompt_text = config["prompt"]
    draw.text((30, prompt_y), prompt_text, font=font_large, fill=config["text_color"])
    
    # Blinking cursor
    cursor_x = 30 + draw.textlength(prompt_text, font=font_large) + 5
    if frame_num % 2 == 0:  # Blink every other frame
        draw.rectangle([cursor_x, prompt_y + 2, cursor_x + 10, prompt_y + 18], fill=config["text_color"])
    
    # ===== Apply CRT Effects =====
    img = add_glow(img, radius=1)
    img = create_scanlines(img, intensity=0.2)
    img = create_crt_curve(img)
    
    # Add slight noise/flicker
    if frame_num % 3 == 0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.97)
    
    return img


def generate_crt_gif(image_path, output_path, config):
    """Generate the full CRT-style GIF animation."""
    print(f"🖼️  Loading image: {image_path}")
    
    # Convert image to ASCII art
    ascii_art = image_to_ascii(image_path, width=config["ascii_width"])
    print(f"📝 Generated ASCII art ({len(ascii_art.split(chr(10)))} lines)")
    
    # Generate frames
    frames = []
    for i in range(config["frames"]):
        print(f"🎬 Generating frame {i + 1}/{config['frames']}...")
        frame = create_frame(ascii_art, config, i, config["frames"])
        frames.append(frame)
    
    # Save as GIF
    print(f"💾 Saving GIF: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=config["frame_duration"],
        loop=0,
        optimize=True
    )
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"✅ Done! File size: {file_size:.1f} KB")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_image = os.path.join(project_root, "images", "girl.jpg")
    output_gif = os.path.join(project_root, "assets", "crt_banner.gif")
    
    if not os.path.exists(input_image):
        print(f"❌ Error: Image not found: {input_image}")
        return
    
    generate_crt_gif(input_image, output_gif, CONFIG)


if __name__ == "__main__":
    main()
