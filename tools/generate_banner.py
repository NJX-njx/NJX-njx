import random
import math

def generate_tech_banner(filename):
    width = 1500
    height = 500
    # GitHub Dark Dimmed Background
    bg_color = "#0d1117" 
    # Tech Accents: Cyan, Purple, Blue
    colors = ["#58a6ff", "#bc8cff", "#3fb950", "#2f81f7"]
    
    svg_content = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="100%" height="100%" fill="{bg_color}"/>',
        # Grid background
        '<defs>',
        '<pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">',
        f'<path d="M 50 0 L 0 0 0 50" fill="none" stroke="#30363d" stroke-width="1"/>',
        '</pattern>',
        '</defs>',
        '<rect width="100%" height="100%" fill="url(#grid)" opacity="0.3"/>'
    ]

    # Generate Neural Nodes and Connections
    nodes = []
    for _ in range(40):
        x = random.randint(100, width - 100)
        y = random.randint(50, height - 50)
        nodes.append((x, y))

    # Draw connections (edges)
    for i, (x1, y1) in enumerate(nodes):
        # Connect to nearest neighbors
        distances = []
        for j, (x2, y2) in enumerate(nodes):
            if i == j: continue
            dist = math.hypot(x2 - x1, y2 - y1)
            distances.append((dist, x2, y2))
        
        distances.sort()
        # Connect top 3 closest
        for dist, x2, y2 in distances[:3]:
            if dist < 250:
                opacity = (1 - dist/250) * 0.5
                stroke_color = random.choice(colors)
                svg_content.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke_color}" stroke-width="1" opacity="{opacity}"/>')

    # Draw nodes
    for x, y in nodes:
        color = random.choice(colors)
        radius = random.randint(2, 5)
        svg_content.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" opacity="0.8"/>')
        # Glow effect
        svg_content.append(f'<circle cx="{x}" cy="{y}" r="{radius*2}" fill="{color}" opacity="0.2"/>')

    # Main Text: NJX
    # Centered, Modern Font
    svg_content.append(f'''
    <text x="50%" y="45%" dominant-baseline="middle" text-anchor="middle" 
          font-family="'Segoe UI', Ubuntu, Sans-Serif" font-weight="bold" font-size="80" fill="#ffffff" letter-spacing="10">
        NJX
    </text>
    <text x="50%" y="60%" dominant-baseline="middle" text-anchor="middle" 
          font-family="'SF Mono', 'Fira Code', monospace" font-size="24" fill="#8b949e" letter-spacing="2">
        ARCHITECTING INTELLIGENCE
    </text>
    ''')

    # Decorative "System Status" lines
    svg_content.append(f'<rect x="100" y="{height-40}" width="200" height="2" fill="#3fb950" opacity="0.8"/>')
    svg_content.append(f'<text x="100" y="{height-20}" font-family="monospace" font-size="12" fill="#3fb950">SYSTEM: ONLINE</text>')
    
    svg_content.append(f'<rect x="{width-300}" y="{height-40}" width="200" height="2" fill="#58a6ff" opacity="0.8"/>')
    svg_content.append(f'<text x="{width-300}" y="{height-20}" font-family="monospace" font-size="12" fill="#58a6ff" text-anchor="end">MODE: RESEARCH</text>')

    svg_content.append('</svg>')

    with open(filename, 'w') as f:
        f.write('\n'.join(svg_content))

if __name__ == "__main__":
    generate_tech_banner("assets/header_v2.svg")
