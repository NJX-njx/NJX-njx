import random
import math

def generate_pro_banner(filename):
    width = 1500
    height = 600
    # GitHub Dark Palette
    bg_color = "#0d1117"
    border_color = "#30363d"
    text_primary = "#c9d1d9"
    text_secondary = "#8b949e"
    accent_blue = "#58a6ff"
    accent_green = "#3fb950"
    accent_purple = "#bc8cff"
    
    svg_content = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: {bg_color};">',
        '<defs>',
        # Gradient for the glow
        '<linearGradient id="glow" x1="0%" y1="0%" x2="100%" y2="0%">',
        f'<stop offset="0%" style="stop-color:{accent_blue};stop-opacity:0" />',
        f'<stop offset="50%" style="stop-color:{accent_blue};stop-opacity:0.5" />',
        f'<stop offset="100%" style="stop-color:{accent_purple};stop-opacity:0" />',
        '</linearGradient>',
        # Pattern for grid
        '<pattern id="smallGrid" width="20" height="20" patternUnits="userSpaceOnUse">',
        f'<path d="M 20 0 L 0 0 0 20" fill="none" stroke="{border_color}" stroke-width="0.5" opacity="0.3"/>',
        '</pattern>',
        '</defs>',
        
        # Background Grid
        f'<rect width="100%" height="100%" fill="url(#smallGrid)" />',
        
        # Decorative "Code" Blocks in background (Abstract)
        f'<rect x="50" y="50" width="300" height="150" rx="10" fill="{border_color}" opacity="0.2" />',
        f'<rect x="{width-350}" y="{height-200}" width="300" height="150" rx="10" fill="{border_color}" opacity="0.2" />',
    ]

    # Generate "Data Stream" Lines (Bezier Curves)
    for i in range(5):
        y_start = height / 2 + (i - 2) * 40
        path_d = f"M 0 {y_start} "
        
        # Create a wave
        for x in range(0, width + 100, 100):
            y_noise = random.randint(-30, 30)
            path_d += f"L {x} {y_start + y_noise} "
            
        opacity = 0.1 + (i * 0.05)
        color = accent_blue if i % 2 == 0 else accent_purple
        svg_content.append(f'<path d="{path_d}" stroke="{color}" stroke-width="2" fill="none" opacity="{opacity}" />')

    # Central "Glass" Card
    card_width = 800
    card_height = 300
    card_x = (width - card_width) / 2
    card_y = (height - card_height) / 2
    
    svg_content.append(f'''
    <!-- Glass Card -->
    <rect x="{card_x}" y="{card_y}" width="{card_width}" height="{card_height}" rx="15" 
          fill="#161b22" stroke="{border_color}" stroke-width="1" opacity="0.9" />
    
    <!-- Top Bar of Card -->
    <path d="M {card_x} {card_y+40} L {card_x+card_width} {card_y+40}" stroke="{border_color}" stroke-width="1" />
    <circle cx="{card_x+20}" cy="{card_y+20}" r="6" fill="#ff5f56" />
    <circle cx="{card_x+45}" cy="{card_y+20}" r="6" fill="#ffbd2e" />
    <circle cx="{card_x+70}" cy="{card_y+20}" r="6" fill="#27c93f" />
    
    <!-- Title in Card Header -->
    <text x="{card_x + card_width/2}" y="{card_y+25}" text-anchor="middle" font-family="monospace" font-size="14" fill="{text_secondary}">
        njx_profile.config
    </text>

    <!-- Main Content Text -->
    <text x="{width/2}" y="{height/2 - 20}" text-anchor="middle" 
          font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif" 
          font-weight="800" font-size="72" fill="{text_primary}" letter-spacing="-1">
        NJX
    </text>
    
    <text x="{width/2}" y="{height/2 + 40}" text-anchor="middle" 
          font-family="monospace" font-size="20" fill="{accent_blue}" letter-spacing="4">
        ARCHITECTING INTELLIGENCE
    </text>
    
    <!-- Decorative Tags -->
    <rect x="{width/2 - 160}" y="{height/2 + 80}" width="100" height="24" rx="12" fill="{border_color}" opacity="0.5"/>
    <text x="{width/2 - 110}" y="{height/2 + 96}" text-anchor="middle" font-family="monospace" font-size="12" fill="{accent_green}">● RESEARCH</text>
    
    <rect x="{width/2 - 50}" y="{height/2 + 80}" width="100" height="24" rx="12" fill="{border_color}" opacity="0.5"/>
    <text x="{width/2}" y="{height/2 + 96}" text-anchor="middle" font-family="monospace" font-size="12" fill="{accent_blue}">● BUILD</text>
    
    <rect x="{width/2 + 60}" y="{height/2 + 80}" width="100" height="24" rx="12" fill="{border_color}" opacity="0.5"/>
    <text x="{width/2 + 110}" y="{height/2 + 96}" text-anchor="middle" font-family="monospace" font-size="12" fill="{accent_purple}">● SHIP</text>
    ''')

    # Bottom Status Bar
    svg_content.append(f'''
    <line x1="0" y1="{height-2}" x2="{width}" y2="{height-2}" stroke="{accent_blue}" stroke-width="4" />
    <rect x="0" y="{height-4}" width="{width}" height="4" fill="url(#glow)" />
    ''')

    svg_content.append('</svg>')

    with open(filename, 'w') as f:
        f.write('\n'.join(svg_content))

if __name__ == "__main__":
    generate_pro_banner("assets/header_pro.svg")
