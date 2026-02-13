"""
Generate a retro CRT terminal style GIF banner (exactly like yetone's profile)
Key features to match yetone:
- Bright green phosphor scanline background (not dark)
- Large ASCII art taking up left half
- Neofetch-style info on right
- White/bright text for ASCII highlights
- CRT curvature and glow
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ============ CONFIGURATION ============
CONFIG = {
    # Personal info (neofetch style) - matching yetone's format exactly
    "name": "NJX",
    "age": "22",
    "work": "BIT · CS",
    "os": "macOS",
    "editor": "VS Code",
    "languages": ["Python, Go,", "TypeScript, Rust,", "C++, Bash"],
    "skills": ["Vision, LLM,", "Agents, Infra"],
    
    # Terminal prompt
    "prompt": "njx@mbp$",
    
    # Dimensions
    "width": 1200,
    "height": 700,
    "ascii_width": 55,
    
    # Animation
    "frames": 6,
    "frame_duration": 250,
}

# ASCII characters from dark to light (reversed for bright = white effect)
ASCII_CHARS = " .,:;i1tfLCG08@"


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
                pass
    return ImageFont.load_default()


def image_to_ascii_with_brightness(image_path, width=55):
    """Convert image to ASCII art, also return brightness values for coloring."""
    img = Image.open(image_path)
    
    # Calculate height (chars are ~2x taller than wide)
    aspect_ratio = img.height / img.width
    height = int(width * aspect_ratio * 0.55)
    
    # Resize and convert to grayscale
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img_gray = img.convert("L")
    
    pixels = list(img_gray.getdata())
    
    ascii_lines = []
    brightness_lines = []
    
    for row in range(height):
        line = ""
        brightness_row = []
        for col in range(width):
            pixel = pixels[row * width + col]
            # Map pixel to ASCII char
            char_idx = int(pixel / 255 * (len(ASCII_CHARS) - 1))
            line += ASCII_CHARS[char_idx]
            brightness_row.append(pixel)
        ascii_lines.append(line)
        brightness_lines.append(brightness_row)
    
    return ascii_lines, brightness_lines


def create_crt_background(width, height):
    """Create the bright green CRT scanline background like yetone's."""
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
    # Create horizontal scanlines with bright green
    for y in range(height):
        for x in range(width):
            # Alternating bright/dark green lines (every 3 pixels)
            if y % 3 == 0:
                # Dark line (gap between phosphor lines)
                pixels[x, y] = (15, 40, 15)
            elif y % 3 == 1:
                # Bright phosphor line
                pixels[x, y] = (45, 140, 45)
            else:
                # Medium line
                pixels[x, y] = (35, 110, 35)
    
    return img


def add_crt_curvature(img):
    """Add CRT screen curvature (darker edges)."""
    width, height = img.size
    pixels = img.load()
    
    center_x, center_y = width / 2, height / 2
    max_dist = (center_x ** 2 + center_y ** 2) ** 0.5
    
    for y in range(height):
        for x in range(width):
            # Distance from center
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            # Vignette factor (stronger at edges)
            factor = 1 - (dist / max_dist) ** 1.5 * 0.5
            factor = max(0.4, factor)
            
            r, g, b = pixels[x, y]
            pixels[x, y] = (int(r * factor), int(g * factor), int(b * factor))
    
    return img


def draw_color_blocks(draw, x, y, block_size=25, gap=3):
    """Draw neofetch-style color blocks (muted CRT colors)."""
    colors = [
        (60, 60, 60),      # Dark
        (180, 100, 80),    # Brownish red
        (80, 180, 80),     # Green
        (180, 180, 80),    # Yellow
        (80, 100, 180),    # Blue (muted)
        (160, 80, 160),    # Magenta (muted)
        (80, 180, 160),    # Cyan (muted)
        (200, 200, 200),   # Light
    ]
    
    for i, color in enumerate(colors):
        x1 = x + i * (block_size + gap)
        draw.rectangle([x1, y, x1 + block_size, y + block_size], fill=color)


def create_frame(ascii_lines, brightness_lines, config, frame_num):
    """Create a single frame matching yetone's style exactly."""
    width = config["width"]
    height = config["height"]
    
    # Create CRT background with scanlines
    img = create_crt_background(width, height)
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    font_ascii = load_font(11)  # Small for ASCII art
    font_info = load_font(22)   # Larger for info text
    font_prompt = load_font(20)
    
    # ===== Draw ASCII Art (Left Side) =====
    ascii_x = 50
    ascii_y = 50
    char_width = 8
    line_height = 12
    
    for row_idx, (line, brightness_row) in enumerate(zip(ascii_lines, brightness_lines)):
        for col_idx, (char, brightness) in enumerate(zip(line, brightness_row)):
            if char.strip():  # Skip spaces
                x = ascii_x + col_idx * char_width
                y = ascii_y + row_idx * line_height
                
                # Color based on brightness - bright areas are white/cream
                if brightness > 200:
                    color = (255, 255, 245)  # White/cream for highlights
                elif brightness > 150:
                    color = (220, 240, 220)  # Light green-white
                elif brightness > 100:
                    color = (150, 220, 150)  # Medium bright green
                elif brightness > 50:
                    color = (80, 160, 80)    # Medium green
                else:
                    color = (50, 120, 50)    # Dark green
                
                draw.text((x, y), char, font=font_ascii, fill=color)
    
    # ===== Draw Info Panel (Right Side) - exactly like yetone =====
    info_x = 650
    info_y = 80
    line_spacing = 42
    
    # Colors for text
    label_color = (100, 200, 100)   # Green for labels
    value_color = (220, 255, 220)   # Bright for values
    
    # Info items matching yetone's format exactly
    info_items = [
        ("Name:", config["name"]),
        ("Age:", config["age"]),
        ("Work:", config["work"]),
        ("OS:", config["os"]),
        ("Editor:", config["editor"]),
    ]
    
    current_y = info_y
    
    for label, value in info_items:
        draw.text((info_x, current_y), label, font=font_info, fill=label_color)
        label_width = draw.textlength(label, font=font_info)
        draw.text((info_x + label_width + 10, current_y), value, font=font_info, fill=value_color)
        current_y += line_spacing
    
    # Languages (multi-line)
    draw.text((info_x, current_y), "Languages:", font=font_info, fill=label_color)
    label_width = draw.textlength("Languages:", font=font_info)
    lang_x = info_x + label_width + 10
    for i, lang_line in enumerate(config["languages"]):
        if i == 0:
            draw.text((lang_x, current_y), lang_line, font=font_info, fill=value_color)
        else:
            # Indent continuation lines
            draw.text((lang_x + 40, current_y + i * 30), lang_line, font=font_info, fill=value_color)
    current_y += line_spacing + (len(config["languages"]) - 1) * 30
    
    # Skills (multi-line)
    draw.text((info_x, current_y), "Skills:", font=font_info, fill=label_color)
    label_width = draw.textlength("Skills:", font=font_info)
    skill_x = info_x + label_width + 10
    for i, skill_line in enumerate(config["skills"]):
        if i == 0:
            draw.text((skill_x, current_y), skill_line, font=font_info, fill=value_color)
        else:
            draw.text((skill_x + 40, current_y + i * 30), skill_line, font=font_info, fill=value_color)
    current_y += line_spacing + (len(config["skills"]) - 1) * 30
    
    # Color blocks
    draw_color_blocks(draw, info_x, current_y + 20)
    
    # ===== Terminal Prompt at Bottom =====
    prompt_y = height - 60
    prompt_text = config["prompt"]
    draw.text((50, prompt_y), prompt_text, font=font_prompt, fill=(100, 200, 100))
    
    # Blinking cursor
    cursor_x = 50 + draw.textlength(prompt_text, font=font_prompt) + 8
    if frame_num % 2 == 0:
        draw.rectangle([cursor_x, prompt_y + 3, cursor_x + 12, prompt_y + 22], fill=(150, 255, 150))
    
    # Apply CRT curvature (darker edges)
    img = add_crt_curvature(img)
    
    # Add slight glow effect
    glow = img.filter(ImageFilter.GaussianBlur(1))
    img = Image.blend(img, glow, 0.15)
    
    return img


def generate_crt_gif(image_path, output_path, config):
    """Generate the CRT-style GIF animation like yetone's."""
    print(f"🖼️  Loading image: {image_path}")
    
    # Convert image to ASCII art with brightness info
    ascii_lines, brightness_lines = image_to_ascii_with_brightness(
        image_path, 
        width=config["ascii_width"]
    )
    print(f"📝 Generated ASCII art ({len(ascii_lines)} lines x {len(ascii_lines[0])} chars)")
    
    # Generate frames
    frames = []
    for i in range(config["frames"]):
        print(f"🎬 Generating frame {i + 1}/{config['frames']}...")
        frame = create_frame(ascii_lines, brightness_lines, config, i)
        frames.append(frame)
    
    # Save as GIF
    print(f"💾 Saving GIF: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=config["frame_duration"],
        loop=0,
        optimize=False  # Better quality
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
