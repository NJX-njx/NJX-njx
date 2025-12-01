import math

def generate_interactive_banner(filename):
    width = 1200
    height = 500
    bg_color = "#0d1117"
    
    # Cyberpunk / Neon Palette
    colors = {
        'input': '#FF6B6B',      # Red
        'hidden': '#4ECDC4',     # Cyan
        'attention': '#FFE66D',  # Yellow
        'output': '#1A535C',     # Dark Blue
        'link': '#58a6ff',       # Blue
        'text': '#c9d1d9',
        'glow': '#bc8cff'        # Purple
    }

    svg_content = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: {bg_color};">',
        '<defs>',
        '<style>',
        '''
            @keyframes flow {
                0% { stroke-dashoffset: 100; }
                100% { stroke-dashoffset: 0; }
            }
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-5px); }
                100% { transform: translateY(0px); }
            }
            .node { transition: all 0.3s ease; }
            .connection { stroke-dasharray: 5; animation: flow 2s linear infinite; }
            .label-text { font-family: monospace; font-weight: bold; }
        ''',
        '</style>',
        '<filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feGaussianBlur stdDeviation="3" result="blur" />',
        '<feComposite in="SourceGraphic" in2="blur" operator="over" />',
        '</filter>',
        '</defs>'
    ]

    # Isometric Projection Helper
    def project(x, y, z):
        scale = 35
        # Center the projection in the SVG
        center_x = width / 2
        center_y = height / 2
        
        # Isometric transformation
        iso_x = (x - z) * math.cos(math.radians(30)) * scale + center_x
        iso_y = (x + z) * math.sin(math.radians(30)) * scale - y * scale + center_y
        return iso_x, iso_y

    def draw_cube(x, y, z, color, label=None, delay=0):
        px, py = project(x, y, z)
        size = 18
        
        # Cube Faces
        p1 = (px, py - size)
        p2 = (px + size * 0.866, py - size * 0.5)
        p3 = (px, py)
        p4 = (px - size * 0.866, py - size * 0.5)
        p5 = (px + size * 0.866, py + size * 0.5)
        p6 = (px, py + size)
        p7 = (px - size * 0.866, py + size * 0.5)

        # Group for animation
        anim_style = f'style="animation: float 3s ease-in-out infinite; animation-delay: {delay}s;"'
        
        cube_svg = f'<g class="node" {anim_style}>'
        
        # Draw faces
        cube_svg += f'<path d="M{p1[0]},{p1[1]} L{p2[0]},{p2[1]} L{p3[0]},{p3[1]} L{p4[0]},{p4[1]} Z" fill="{color}" fill-opacity="0.9" stroke="{bg_color}" stroke-width="1"/>'
        cube_svg += f'<path d="M{p3[0]},{p3[1]} L{p2[0]},{p2[1]} L{p5[0]},{p5[1]} L{p6[0]},{p6[1]} Z" fill="{color}" fill-opacity="0.7" stroke="{bg_color}" stroke-width="1"/>'
        cube_svg += f'<path d="M{p4[0]},{p4[1]} L{p3[0]},{p3[1]} L{p6[0]},{p6[1]} L{p7[0]},{p7[1]} Z" fill="{color}" fill-opacity="0.5" stroke="{bg_color}" stroke-width="1"/>'
        
        if label:
             # Label with background for readability
             cube_svg += f'<rect x="{px-40}" y="{py-size*2.5}" width="80" height="20" rx="4" fill="#000" opacity="0.7" />'
             cube_svg += f'<text x="{px}" y="{py-size*2.5+14}" text-anchor="middle" class="label-text" font-size="10" fill="white">{label}</text>'

        cube_svg += '</g>'
        svg_content.append(cube_svg)
        return px, py

    # --- Draw Architecture ---

    # 1. Input Layer (Bottom Left)
    # Shifted coordinates: x=-2..2 -> x=-5..-1 (shift -3), z=7 (move left/forward)
    input_coords = []
    for i in range(-2, 3):
        # Position: Lower left in isometric space
        # x = i - 3, z = 7
        px, py = draw_cube(i - 3, -2, 7, colors['input'], delay=i*0.1)
        input_coords.append((px, py))
    
    # Text slightly nudged left relative to center cube
    svg_content.append(f'<text x="{input_coords[2][0] - 20}" y="{input_coords[2][1]+50}" text-anchor="middle" class="label-text" font-size="12" fill="{colors["input"]}">INPUT: DATA</text>')

    # 2. Hidden Layers (Center)
    # Centered at y=0
    hidden_coords = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i+j) % 2 == 0:
                px, py = draw_cube(i, 0, j, colors['hidden'], delay=(i+j)*0.1)
                hidden_coords.append((px, py))

    # 3. Output Layer (Top Right)
    # Shifted coordinates: x=-2..2 -> x=1..5 (shift +3), z=-7 (move right/back), y=1
    output_coords = []
    for i in range(-2, 3):
        # Position: Upper right in isometric space
        # x = i + 3, z = -7, y = 1
        px, py = draw_cube(i + 3, 1, -7, colors['attention'], delay=i*0.1)
        output_coords.append((px, py))
        
    # Text slightly nudged right relative to center cube
    svg_content.append(f'<text x="{output_coords[2][0] + 20}" y="{output_coords[2][1]-40}" text-anchor="middle" class="label-text" font-size="12" fill="{colors["attention"]}">OUTPUT: INTELLIGENCE</text>')

    # --- Draw Animated Connections ---
    # Input to Hidden
    for px, py in input_coords:
        target = hidden_coords[len(hidden_coords)//2]
        svg_content.append(f'<line x1="{px}" y1="{py}" x2="{target[0]}" y2="{target[1]}" stroke="{colors["link"]}" stroke-width="1" class="connection" opacity="0.4" />')

    # Hidden to Output
    for px, py in hidden_coords:
        target = output_coords[len(output_coords)//2]
        svg_content.append(f'<line x1="{px}" y1="{py}" x2="{target[0]}" y2="{target[1]}" stroke="{colors["glow"]}" stroke-width="1" class="connection" opacity="0.4" />')

    # --- HUD Elements ---
    svg_content.append(f'''
    <g transform="translate(50, 50)">
        <rect width="200" height="80" fill="none" stroke="{colors['link']}" stroke-width="1" />
        <text x="10" y="25" class="label-text" font-size="14" fill="{colors['link']}">SYSTEM: NJX-KERNEL</text>
        <text x="10" y="45" class="label-text" font-size="10" fill="{colors['text']}">> STATUS: ONLINE</text>
        <text x="10" y="60" class="label-text" font-size="10" fill="{colors['text']}">> LOAD: 98%</text>
        
        <!-- Animated Blinking Cursor -->
        <rect x="10" y="70" width="8" height="2" fill="{colors['link']}">
            <animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite" />
        </rect>
    </g>
    ''')

    svg_content.append('</svg>')

    with open(filename, 'w') as f:
        f.write('\n'.join(svg_content))

if __name__ == "__main__":
    generate_interactive_banner("assets/header_interactive.svg")
