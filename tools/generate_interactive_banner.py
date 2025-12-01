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
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background-color: {bg_color};">',
        '<defs>',
        '<style>',
        '''
            @keyframes flow {
                0% { stroke-dashoffset: 100; }
                100% { stroke-dashoffset: 0; }
            }
            @keyframes pulse {
                0% { opacity: 0.6; transform: scale(1); }
                50% { opacity: 1; transform: scale(1.05); }
                100% { opacity: 0.6; transform: scale(1); }
            }
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-5px); }
                100% { transform: translateY(0px); }
            }
            .node { cursor: pointer; transition: all 0.3s ease; }
            .node:hover { filter: brightness(1.3); }
            .node:hover rect, .node:hover path { stroke: white; stroke-width: 2px; }
            .connection { stroke-dasharray: 5; animation: flow 2s linear infinite; }
            .label-text { font-family: 'Courier New', monospace; font-weight: bold; pointer-events: none; }
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
        iso_x = (x - z) * math.cos(math.radians(30)) * scale + width / 2
        iso_y = (x + z) * math.sin(math.radians(30)) * scale - y * scale + height / 2 + 50
        return iso_x, iso_y

    def draw_cube(x, y, z, color, link_id=None, label=None, delay=0):
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

        # Group for interactivity and animation
        anim_style = f'style="animation: float 3s ease-in-out infinite; animation-delay: {delay}s;"'
        
        cube_svg = f'<g class="node" {anim_style}>'
        
        if link_id:
            cube_svg = f'<a xlink:href="#{link_id}">' + cube_svg

        # Draw faces
        cube_svg += f'<path d="M{p1[0]},{p1[1]} L{p2[0]},{p2[1]} L{p3[0]},{p3[1]} L{p4[0]},{p4[1]} Z" fill="{color}" fill-opacity="0.9" stroke="{bg_color}" stroke-width="1"/>'
        cube_svg += f'<path d="M{p3[0]},{p3[1]} L{p2[0]},{p2[1]} L{p5[0]},{p5[1]} L{p6[0]},{p6[1]} Z" fill="{color}" fill-opacity="0.7" stroke="{bg_color}" stroke-width="1"/>'
        cube_svg += f'<path d="M{p4[0]},{p4[1]} L{p3[0]},{p3[1]} L{p6[0]},{p6[1]} L{p7[0]},{p7[1]} Z" fill="{color}" fill-opacity="0.5" stroke="{bg_color}" stroke-width="1"/>'
        
        if label:
             # Label with background for readability
             cube_svg += f'<rect x="{px-40}" y="{py-size*2.5}" width="80" height="20" rx="4" fill="#000" opacity="0.7" />'
             cube_svg += f'<text x="{px}" y="{py-size*2.5+14}" text-anchor="middle" class="label-text" font-size="10" fill="white">{label}</text>'

        if link_id:
            cube_svg += '</a>'
            
        cube_svg += '</g>'
        svg_content.append(cube_svg)
        return px, py

    # --- Draw Architecture ---

    # 1. Input Layer (Data Ingestion) -> Links to "About Me" / Intro
    input_coords = []
    for i in range(-2, 3):
        px, py = draw_cube(i, 0, 2, colors['input'], link_id="user-content-core-identity", delay=i*0.1)
        input_coords.append((px, py))
    
    svg_content.append(f'<text x="{input_coords[2][0]}" y="{input_coords[2][1]+50}" text-anchor="middle" class="label-text" font-size="12" fill="{colors["input"]}">INPUT: DATA</text>')

    # 2. Hidden Layers (Processing) -> Links to "Research"
    hidden_coords = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Only draw some cubes to make it look sparse/complex
            if (i+j) % 2 == 0:
                px, py = draw_cube(i, 3, j, colors['hidden'], link_id="user-content-research-frontiers", delay=(i+j)*0.1)
                hidden_coords.append((px, py))

    # 3. Output Layer (Results) -> Links to "Projects"
    output_coords = []
    for i in range(-2, 3):
        px, py = draw_cube(i, 6, -2, colors['attention'], link_id="user-content-selected-deployments", delay=i*0.1)
        output_coords.append((px, py))
        
    svg_content.append(f'<text x="{output_coords[2][0]}" y="{output_coords[2][1]-40}" text-anchor="middle" class="label-text" font-size="12" fill="{colors["attention"]}">OUTPUT: INTELLIGENCE</text>')

    # --- Draw Animated Connections ---
    # Input to Hidden
    for px, py in input_coords:
        target = hidden_coords[len(hidden_coords)//2] # Connect to center hidden
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
    
    <g transform="translate({width-250}, 50)">
        <text x="0" y="20" class="label-text" font-size="12" fill="{colors['text']}" text-anchor="end">NAVIGATION_CONTROLS:</text>
        <text x="0" y="40" class="label-text" font-size="10" fill="{colors['input']}" text-anchor="end">[CLICK NODES TO INSPECT]</text>
    </g>
    ''')

    svg_content.append('</svg>')

    with open(filename, 'w') as f:
        f.write('\n'.join(svg_content))

if __name__ == "__main__":
    generate_interactive_banner("assets/header_interactive.svg")
